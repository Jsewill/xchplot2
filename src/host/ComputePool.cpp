#include "host/ComputePool.hpp"
#include "host/GroupHarvesterProfile.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <string_view>
#include <thread>
#include <stdexcept>
#include <utility>

namespace group_grow::pool
{
std::string gpu_recovery_reason(std::string text)
{
    // Driver status is printed on terminals on the owner as well as the worker.
    // Keep a bounded ASCII line even when a peer sends arbitrary diagnostics.
    if (text.size() > 240) text.erase(0, text.size() - 240);
    for (char& c : text) if (static_cast<unsigned char>(c) < 32 || static_cast<unsigned char>(c) > 126) c = ' ';
    return text;
}
void write_worker_hardware(Writer& out, WorkerHardware const& value)
{
    out.string(value.host); out.string(value.gpu_identity);
    out.u32(value.partition.cpu_threads); out.u32(value.partition.slots);
    out.u64(value.partition.host_bytes); out.u64(value.partition.gpu_bytes);
    out.u32(value.logical_cpus); out.u64(value.total_host_bytes); out.u64(value.available_host_bytes); out.u64(value.total_gpu_bytes);
    out.u64(value.standing_host_bytes); out.u64(value.observed_at_unix_ms);
}
WorkerHardware read_worker_hardware(Reader& in)
{
    WorkerHardware value; value.host = in.string(255); value.gpu_identity = in.string(255);
    value.partition.cpu_threads = in.u32(); value.partition.slots = in.u32();
    value.partition.host_bytes = in.u64(); value.partition.gpu_bytes = in.u64();
    value.logical_cpus = in.u32(); value.total_host_bytes = in.u64(); value.available_host_bytes = in.u64(); value.total_gpu_bytes = in.u64();
    value.standing_host_bytes = in.u64(); value.observed_at_unix_ms = in.u64(); return value;
}
namespace
{
using Clock = std::chrono::steady_clock;
constexpr char broker_restarted[] = "broker restarted; open a new owner session";
void require(bool value, char const* message) { if (!value) throw std::runtime_error(message); }
enum class ErrorCode : uint8_t { Invalid = 1, StaleOwner = 2, StaleDelivery = 3, StaleLease = 4 };
enum class BeginResult : uint8_t { Stale, Running, AwaitingAdmission };
enum class Operation : uint8_t { OpenOwner = 1, Acquire, Cancel, Renew, Register, Begin, Finish, Delivered, Reject, Abandon, GenerationHint, Heartbeat, Unregister, CloseOwner, Status, WorkerAdmission, RetireWorker, GenerationAvailability, GenerationRoutes, RegisterWorkerRoute, RegisterGpuRecovery, StatusGpuRecovery, GenerationRoutesGpuRecovery, AcquireFiniteGeneration, FinitePackedRoute, GenerationPaths, DeliveredOnPath };
class UnknownOperation : public std::runtime_error { using std::runtime_error::runtime_error; };
class UnsupportedBacklog : public std::runtime_error { using std::runtime_error::runtime_error; };
constexpr uint8_t independent_flag = 0x80;
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
    writer.u8(uint8_t(task.kind) | (task.independent_members ? independent_flag : 0)); writer.u8(uint8_t(task.generation_output));
    writer.u8(task.k); writer.u8(task.strength); write_resources(writer, task.resources);
    writer.u32(task.chunk_count); writer.u64(task.max_output_bytes);
}
Task read_task(Reader& reader)
{
    Task result; result.task_id = reader.string(256); result.compatibility = reader.string(256);
    result.attempt = reader.u64(); result.sequence = reader.u64(); result.work_units = reader.u64();
    auto const kind = reader.u8(); result.kind = Kind(kind & ~independent_flag); result.independent_members = kind & independent_flag;
    result.generation_output = GenerationOutput(reader.u8());
    require(result.generation_output <= GenerationOutput::Fresh &&
        (!result.independent_members || result.generation_output == GenerationOutput::Contributions) &&
        (result.kind != Kind::Pack || result.generation_output == GenerationOutput::Packed), "invalid generation output");
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
    writer.string(offer.instance); writer.string(offer.compatibility); writer.string(offer.codec_compatibility);
    writer.string(offer.producer_build); write_endpoint(writer, offer.endpoint);
    write_resources(writer, offer.capacity); writer.u64(offer.capacity_revision); writer.u32(offer.generation_threads); writer.u32(offer.capabilities); writer.u8(offer.min_k); writer.u8(offer.max_k);
    write_profiles(writer, offer.generation_profiles);
    writer.u32(offer.codec_limits.max_chunks); writer.u64(offer.codec_limits.max_task_bytes); writer.u8(uint8_t(offer.admission));
    write_worker_hardware(writer, offer.hardware); writer.u32(offer.independent_k_mask);
}
WorkerOffer read_offer(Reader& reader)
{
    WorkerOffer result; result.instance = reader.string(64); result.compatibility = reader.string(256);
    result.codec_compatibility = reader.string(256); result.producer_build = reader.string(128);
    result.endpoint = read_endpoint(reader); result.capacity = read_resources(reader); result.capacity_revision = reader.u64(); result.generation_threads = reader.u32(); result.capabilities = reader.u32();
    result.min_k = reader.u8(); result.max_k = reader.u8(); result.generation_profiles = read_profiles(reader);
    result.codec_limits.max_chunks = reader.u32(); result.codec_limits.max_task_bytes = reader.u64();
    result.admission = WorkerAdmission(reader.u8()); result.hardware = read_worker_hardware(reader);
    result.independent_k_mask = reader.u32(); return result;
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
    // Each request records unsupported operations against the broker it queried.
    // An older reply in flight must not disable them on the replacement.
    struct GenerationApi
    {
        std::atomic<bool> routes{true}, backlog{true}, gpu_recovery{true}, finite{true}, paths{true};
    };
    std::atomic<std::shared_ptr<GenerationApi>> generation_api{std::make_shared<GenerationApi>()};
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
    connections_->generation_api.store(std::make_shared<Connections::GenerationApi>());
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
        if (code == ErrorCode::StaleOwner && message == broker_restarted) throw BrokerRestarted("compute broker: " + message);
        if (code == ErrorCode::StaleOwner) throw StaleOwnerSession("compute broker: " + message);
        if (code == ErrorCode::StaleDelivery) throw StaleDelivery("compute broker: " + message);
        if (code == ErrorCode::StaleLease) throw StaleLease("compute broker: " + message);
        if (message == "unknown broker operation") throw UnknownOperation("compute broker: " + message);
        if (message == "invalid generation backlog") throw UnsupportedBacklog("compute broker: " + message);
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
    Reader reader(bytes); auto result = read_session(reader); reader.finish();
    connections_->generation_api.store(std::make_shared<Connections::GenerationApi>()); return result;
}
void BrokerClient::close_owner(OwnerSession const& session)
{ auto writer = request(Operation::CloseOwner); write_session(writer, session); auto bytes = call(writer); Reader(bytes).finish(); }
std::optional<Lease> BrokerClient::acquire(OwnerSession const& owner, Task const& task, Clock::time_point other_work_until)
{
    auto const now = Clock::now();
    auto const api = connections_->generation_api.load();
    bool const finite = other_work_until != Clock::time_point{} && other_work_until <= now && task.kind != Kind::Pack &&
        (task.generation_output == GenerationOutput::Packed || task.generation_output == GenerationOutput::Fresh) &&
        api->finite.load();
    auto writer = request(finite ? Operation::AcquireFiniteGeneration : Operation::Acquire);
    write_session(writer, owner); write_task(writer, task);
    uint32_t remaining = 0, backlog = 0;
    if (tls_.deadline != Clock::time_point::max())
    {
        auto const left = std::chrono::ceil<std::chrono::milliseconds>(tls_.deadline - now).count();
        if (left <= 0) throw TransientTransport("broker control request timed out");
        remaining = uint32_t(std::min<int64_t>(left, UINT32_MAX));
        if (task.kind != Kind::Pack && (task.independent_members || api->backlog.load()) && other_work_until > now)
        {
            auto const ahead = std::chrono::duration_cast<std::chrono::milliseconds>(other_work_until - now).count();
            backlog = uint32_t(std::min<int64_t>(remaining, ahead));
        }
    }
    writer.u32(remaining); writer.u32(backlog);
    std::string peer, node; std::vector<uint8_t> bytes;
    try { bytes = call(writer, &peer, &node); }
    catch (UnknownOperation const&)
    {
        if (!finite) throw;
        api->finite.store(false); return acquire(owner, task, other_work_until);
    }
    catch (UnsupportedBacklog const&)
    {
        // Older protocol-8 brokers reject this optional field before opening
        // or granting the job. Only that explicit rejection permits fallback.
        if (task.independent_members || !backlog) throw;
        api->backlog.store(false); return acquire(owner, task);
    }
    Reader reader(bytes); auto const present = reader.u8(); require(present <= 1, "invalid broker grant flag");
    std::optional<Lease> result; if (present)
    {
        result = read_lease(reader);
        require(result->broker_identity == peer, "lease authority differs from authenticated broker");
        require(node.empty() || result->broker_node_id == node, "lease node differs from authenticated broker");
    }
    reader.finish(); return result;
}
bool BrokerClient::generation_paths(OwnerSession const& owner, GenerationPaths const& paths)
{
    auto const api = connections_->generation_api.load();
    if (!api->paths.load()) return false;
    require(paths.size() <= max_status_page_size, "excessive generation paths");
    auto writer = request(Operation::GenerationPaths); write_session(writer, owner); writer.u8(uint8_t(paths.size()));
    for (auto const& [worker, path] : paths) { writer.string(worker); writer.string(path); }
    try { auto bytes = call(writer); Reader(bytes).finish(); }
    catch (UnknownOperation const&) { api->paths.store(false); return false; }
    return true;
}
std::optional<std::pair<std::string, Kind>> BrokerClient::finite_packed_route(OwnerSession const& owner, Task const& task)
{
    auto const api = connections_->generation_api.load();
    if (!api->finite.load()) return {};
    auto writer = request(Operation::FinitePackedRoute); write_session(writer, owner); write_task(writer, task);
    std::vector<uint8_t> bytes;
    try { bytes = call(writer); }
    catch (UnknownOperation const&) { api->finite.store(false); return {}; }
    Reader reader(bytes); auto const identity = reader.string(64);
    if (identity.empty()) { reader.finish(); return {}; }
    auto const kind = Kind(reader.u8()); reader.finish();
    require(identifier(identity, 64) && (kind == Kind::GenerateCpu || kind == Kind::GenerateGpu), "invalid finite generation route");
    return std::pair{identity, kind};
}
uint32_t BrokerClient::generation_hint(OwnerSession const& owner, Task const& task, uint32_t* slots,
    GenerationAvailability* availability, std::string const& codec, std::shared_ptr<GenerationRoster const>* roster)
{
    require(!roster || availability, "worker roster requires a detailed capacity hint");
    auto query = [&](Operation op)
    {
        auto writer = request(op); write_session(writer, owner); write_task(writer, task);
        if (availability) writer.string(codec);
        return call(writer);
    };
    auto const api = connections_->generation_api.load();
    bool routes = roster && api->routes.load();
    std::vector<uint8_t> bytes;
    bool recovery = routes && api->gpu_recovery.load();
    if (recovery) try { bytes = query(Operation::GenerationRoutesGpuRecovery); }
    catch (UnknownOperation const&) { api->gpu_recovery.store(false); recovery = false; }
    if (routes && !recovery) try { bytes = query(Operation::GenerationRoutes); }
    catch (UnknownOperation const&)
    {
        // Older compatible brokers still supply the aggregate capacity. Only
        // an explicitly unknown advisory operation selects this fallback.
        api->routes.store(false); routes = false;
    }
    if (!routes) bytes = query(availability ? Operation::GenerationAvailability : Operation::GenerationHint);
    Reader reader(bytes); auto const result = reader.u32(), parallel = reader.u32();
    if (availability)
    {
        availability->admission.reset();
        availability->independent_batch = reader.u32();
        auto const available = reader.u8();
        require(availability->independent_batch <= max_generation_batch_members &&
            (available == 0 || available == 1 || available == 3), "invalid worker availability");
        availability->codec_available = available & 1;
        availability->concurrent_codec = available & 2;
        auto previous = parallel;
        for (auto& count : std::span(availability->independent_slots).subspan(1))
        {
            count = reader.u32();
            require(count <= previous, "invalid independent generation slot count");
            previous = count;
        }
    }
    if (roster)
    {
        roster->reset();
        if (routes)
        {
            auto snapshot = std::make_shared<GenerationRoster>();
            auto const complete = reader.u8(), count = reader.u8();
            require(complete <= 1 && count <= max_status_page_size && (complete || count == max_status_page_size),
                "invalid generation roster size");
            snapshot->complete = complete; snapshot->routes.reserve(count);
            for (unsigned i = 0; i < count; ++i)
            {
                GenerationRoute route; route.identity = reader.string(64); route.instance = reader.string(64);
                route.endpoint = read_endpoint(reader); route.host = reader.string(255); route.gpu_identity = reader.string(255);
                route.cpu_slots = reader.u32(); route.gpu_slots = reader.u32();
                require(identifier(route.identity, 64) && identifier(route.instance, 64) &&
                    !route.endpoint.host.empty() && route.endpoint.port && (route.cpu_slots || route.gpu_slots) &&
                    (snapshot->routes.empty() || snapshot->routes.back().identity < route.identity), "invalid generation route");
                snapshot->routes.push_back(std::move(route));
            }
            GenerationAdmission admission;
            admission.registered = reader.u32(); admission.eligible = reader.u32();
            admission.available_slots = reader.u32(); admission.other_owner_tasks = reader.u32();
            uint64_t waiting = 0;
            for (auto& number : admission.waiting) { number = reader.u32(); waiting += number; }
            require(admission.eligible <= admission.registered && admission.available_slots <= parallel &&
                count == std::min<uint32_t>(admission.eligible, max_status_page_size) &&
                snapshot->complete == (admission.eligible == count) &&
                waiting >= admission.registered - admission.eligible && waiting <= admission.registered,
                "invalid generation admission");
            if (recovery)
            {
                auto count = reader.u8(); require(count <= max_status_page_size && count <= admission.registered,
                    "invalid GPU recovery count");
                while (count--) snapshot->recovering_gpus.push_back({reader.string(255), gpu_recovery_reason(reader.string(240))});
            }
            availability->admission = admission;
            *roster = std::move(snapshot);
        }
    }
    reader.finish();
    if (slots) *slots = parallel;
    return result;
}
void BrokerClient::cancel(OwnerSession const& owner, std::string const& task, uint64_t attempt)
{
    auto writer = request(Operation::Cancel); write_session(writer, owner); writer.string(task); writer.u64(attempt);
    auto bytes = call(writer); Reader(bytes).finish();
}
void BrokerClient::renew(Lease const& lease)
{ auto writer = request(Operation::Renew); write_lease(writer, lease); auto bytes = call(writer); Reader(bytes).finish(); }
std::optional<std::pair<std::string, Endpoint>> BrokerClient::register_recovering_gpu(WorkerOffer const& offer,
    bool retain_route, std::string* peer_identity, std::string* peer_node_id)
{
    if (offer.gpu_recovery.empty()) return {};
    auto const api = connections_->generation_api.load();
    if (!api->gpu_recovery.load()) return {};
    auto writer = request(Operation::RegisterGpuRecovery); write_offer(writer, offer);
    writer.u8(retain_route); writer.string(offer.gpu_recovery);
    try
    {
        auto bytes = call(writer, peer_identity, peer_node_id);
        Reader in(bytes); auto epoch = in.string(64); auto endpoint = read_endpoint(in); in.finish();
        require(!epoch.empty() && endpoint.port, "invalid GPU recovery registration");
        return std::pair{std::move(epoch), std::move(endpoint)};
    }
    catch (UnknownOperation const&) { api->gpu_recovery.store(false); return {}; }
}
std::string BrokerClient::register_worker(WorkerOffer const& offer, std::string* peer_identity, std::string* peer_node_id)
{
    if (auto received = register_recovering_gpu(offer, false, peer_identity, peer_node_id)) return received->first;
    auto writer = request(Operation::Register); write_offer(writer, offer);
    auto bytes = call(writer, peer_identity, peer_node_id);
    Reader reader(bytes); auto epoch = reader.string(64); reader.finish();
    require(!epoch.empty(), "broker registration omitted its epoch"); return epoch;
}
std::string BrokerClient::register_worker_route(WorkerOffer& offer, std::string* peer_identity, std::string* peer_node_id)
{
    if (auto received = register_recovering_gpu(offer, true, peer_identity, peer_node_id))
    { offer.endpoint = std::move(received->second); return received->first; }
    auto writer = request(Operation::RegisterWorkerRoute); write_offer(writer, offer);
    try
    {
        auto bytes = call(writer, peer_identity, peer_node_id);
        Reader reader(bytes); auto epoch = reader.string(64); auto endpoint = read_endpoint(reader); reader.finish();
        require(!epoch.empty() && !endpoint.host.empty() && endpoint.port, "broker route registration omitted its endpoint or epoch");
        offer.endpoint = std::move(endpoint); return epoch;
    }
    catch (UnknownOperation const&) {}
    // Older brokers can expose the same worker's current endpoint through
    // authenticated status. The caller's registration deadline bounds every page.
    auto const identity = certificate_identity(tls_.cert_file);
    StatusRequest query;
    for (;;)
    {
        auto page = status(query);
        auto found = std::find_if(page.worker_records.begin(), page.worker_records.end(),
            [&](auto const& worker) { return worker.identity == identity && worker.instance == offer.instance; });
        if (found != page.worker_records.end()) { offer.endpoint = found->endpoint; break; }
        if (!page.next_cursor) break;
        require(page.next_cursor > query.cursor && page.next_cursor <= 4096, "invalid broker worker status cursor");
        query.cursor = page.next_cursor;
    }
    return register_worker(offer, peer_identity, peer_node_id);
}
void BrokerClient::unregister_worker(std::string const& instance)
{ auto writer = request(Operation::Unregister); writer.string(instance); auto bytes = call(writer); Reader(bytes).finish(); }
bool BrokerClient::retire_worker(std::string const& instance)
{
    auto writer = request(Operation::RetireWorker); writer.string(instance); auto bytes = call(writer); Reader reader(bytes);
    auto const idle = reader.u8(); require(idle <= 1, "invalid worker retirement flag"); reader.finish(); return idle;
}
bool BrokerClient::heartbeat(std::string const& instance, uint64_t revision, std::optional<WorkerAdmission> admission)
{
    auto writer = request(Operation::Heartbeat); writer.string(instance); writer.u64(revision);
    writer.u8(admission ? uint8_t(*admission) : 0); auto bytes = call(writer);
    Reader reader(bytes); auto const current = reader.u8(); require(current <= 1, "invalid broker heartbeat flag");
    reader.finish(); return current;
}
bool BrokerClient::begin(Lease const& lease, std::string const& owner, bool* awaiting_admission)
{
    auto writer = request(Operation::Begin); write_lease(writer, lease); writer.string(owner); auto bytes = call(writer);
    Reader reader(bytes); auto const result = BeginResult(reader.u8());
    require(result <= BeginResult::AwaitingAdmission, "invalid broker begin result"); reader.finish();
    if (awaiting_admission) *awaiting_admission = result == BeginResult::AwaitingAdmission;
    return result == BeginResult::Running;
}
void BrokerClient::finish(Lease const& lease, std::optional<CapacityUpdate> const& update)
{
    auto writer = request(Operation::Finish); write_lease(writer, lease); write_update(writer, update);
    auto bytes = call(writer); Reader(bytes).finish();
}
void BrokerClient::delivered(Lease const& lease, DeliveryObservation const& observation, std::string const& generation_path)
{
    auto const api = connections_->generation_api.load();
    bool const scoped = !generation_path.empty() && api->paths.load();
    auto writer = request(scoped ? Operation::DeliveredOnPath : Operation::Delivered);
    write_lease(writer, lease); writer.u64(observation.elapsed_ns); writer.u64(observation.input_bytes); writer.u64(observation.output_bytes);
    if (scoped) writer.string(generation_path);
    try { auto bytes = call(writer); Reader(bytes).finish(); }
    catch (UnknownOperation const&)
    {
        if (!scoped) throw;
        api->paths.store(false); delivered(lease, observation);
    }
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
    BrokerStatusPage result;
    auto page = [&](Operation op)
    {
        auto writer = request(op); writer.u8(uint8_t(query.section)); writer.u64(query.cursor); writer.u32(query.limit);
        return call(writer, &result.identity);
    };
    auto const api = connections_->generation_api.load();
    bool recovery = query.section == StatusSection::Workers && api->gpu_recovery.load();
    std::vector<uint8_t> bytes;
    if (recovery) try { bytes = page(Operation::StatusGpuRecovery); }
    catch (UnknownOperation const&) { api->gpu_recovery.store(false); recovery = false; }
    if (!recovery) bytes = page(Operation::Status);
    Reader in(bytes); result.section = StatusSection(in.u8());
    require(result.section == query.section, "broker status section mismatch");
    result.host = in.string(255);
    result.next_cursor = in.u64(); result.workers = in.u64(); result.owners = in.u64(); result.jobs = in.u64();
    result.queued = in.u64(); result.granted = in.u64(); result.running = in.u64(); result.cancelled = in.u64();
    auto const count = in.u32(); require(count <= query.limit && count <= max_status_page_size, "broker status page exceeds limit");
    for (uint32_t index = 0; index < count; ++index)
    {
        if (result.section == StatusSection::Workers)
        {
            BrokerWorkerStatus value; value.identity = in.string(64); value.instance = in.string(64); value.compatibility = in.string(256);
            value.codec_compatibility = in.string(256); value.producer_build = in.string(128);
            value.capabilities = in.u32();
            value.endpoint = read_endpoint(in);
            value.capacity = read_resources(in); value.reserved = read_resources(in); value.admission = WorkerAdmission(in.u8()); value.seen_ms = in.u64();
            value.hardware = read_worker_hardware(in);
            if (recovery) value.gpu_recovery = gpu_recovery_reason(in.string(240));
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
        GenerationOutput output;
        bool independent;
        struct Sample { DeliveryObservation observation; uint64_t units = 0; Clock::time_point at{}; Resources resources{}; std::string origin; };
        std::array<Sample, 8> samples{};
        size_t count = 0, next = 0;
        uint64_t assigned = 0;
        bool matches(Task const& task, Kind selected, std::string const& source, uint32_t threads, bool framing = true) const
        { return owner == source && kind == selected && k == task.k && strength == task.strength && cpu_threads == threads &&
            output == task.generation_output && (!framing || independent == task.independent_members); }
        long double cost(uint64_t units, bool singletons = false) const
        {
            long double total = 0, same = 0; size_t used = 0, exact = 0;
            for (size_t i = 0; i < count; ++i)
                if (!singletons || samples[i].units == 1)
                {
                    auto const elapsed = samples[i].observation.elapsed_ns;
                    if (samples[i].units == units) { same += elapsed; ++exact; }
                    total += static_cast<long double>(elapsed) / samples[i].units; ++used;
                }
            // A pipeline pays setup once. Its measured duration at this width
            // is more informative than scaling other widths by member count.
            if (exact) return same / exact;
            return used ? total / used * units : std::numeric_limits<long double>::max();
        }
        long double remaining_ns(uint64_t units, long double elapsed, bool strictly_larger = false) const
        {
            uint64_t width = UINT64_MAX, lower = UINT64_MAX, upper = 0;
            size_t observations = 0;
            for (size_t i = 0; i < count; ++i)
            {
                auto const& sample = samples[i];
                // A smaller joint batch also pays for setup and its final
                // delivery. Its first overrun can still fall inside a wider
                // measured batch of the same framing. An ordinary singleton
                // has no joint encoding, so it keeps its own service class.
                if (sample.units < units || (!independent && units == 1 && sample.units != units)) continue;
                if (strictly_larger && sample.units == units) continue;
                if (sample.units < width) { width = sample.units; lower = UINT64_MAX; upper = 0; observations = 0; }
                if (sample.units != width) continue;
                auto const ns = sample.observation.elapsed_ns;
                lower = std::min(lower, ns); upper = std::max(upper, ns); ++observations;
            }
            long double remaining = upper - elapsed;
            // A live HDD flush can run just beyond the slowest recent
            // delivery. Bound its extension by the variation at this width.
            if (remaining <= 0 && width == units && observations >= 3) remaining += upper - lower;
            return remaining;
        }
        void add(DeliveryObservation observation, uint64_t units, Resources const& resources, std::string origin)
        { samples[next] = {observation, units, Clock::now(), resources, std::move(origin)}; next = (next + 1) % samples.size(); count = std::min(count + 1, samples.size()); }
    };
    struct Worker
    {
        WorkerOffer offer;
        Clock::time_point seen;
        std::vector<History> histories;
        Clock::time_point route_since = seen;
    };
    struct Owner
    {
        std::string identity;
        OwnerSession session;
        uint64_t highest_sequence = 0;
        long double service = 0;
        bool ephemeral = false;
        Clock::time_point seen = Clock::now();
        BrokerClient::GenerationPaths paths;
    };
    struct Job
    {
        std::string owner;
        uint64_t status_id = 0;
        uint64_t assignment = 0;
        Task requested;
        std::optional<Lease> lease;
        State state = State::Queued;
        Clock::time_point expires, touched, granted_at, start_by = Clock::time_point::max();
        Clock::time_point other_work_until{};
        Clock::time_point created = Clock::now();
        Clock::time_point unreported_finish{};
        bool observed = false, finite = false;
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
    void release_unstarted(Job& job)
    {
        // A cold batch that never began learned nothing about its worker.
        auto const& lease = *job.lease;
        auto worker = workers.find(lease.worker_identity);
        if (job.requested.work_units > 1 && worker != workers.end() && worker->second.offer.instance == lease.worker_instance)
            for (auto& sample : worker->second.histories)
                if (!sample.count && sample.assigned == job.assignment) sample.assigned = 0;
        job.state = State::Finished;
    }
    void expire()
    {
        auto const now = Clock::now();
        for (auto& [key, job] : jobs)
        {
            // An owner's deadline can also prevent its final Cancel request.
            // Polls keep queued work alive; abandoned requests must become
            // evictable and stop consuming grants when workers return.
            if (((job.state == State::Queued || job.state == State::Granted) && now >= job.start_by) ||
                (reserved(job) && now >= job.expires) ||
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
        if (session.epoch != epoch) throw StaleOwnerSession(broker_restarted);
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
    using Availability = std::map<std::string, Resources>;
    Availability available(std::string const& observer = {}, uint32_t* other_owner_tasks = nullptr) const
    {
        Availability result;
        for (auto const& [identity, worker] : workers) result.emplace(identity, worker.offer.capacity);
        for (auto const& [key, job] : jobs)
            if (reserved(job))
            {
                // Reservations fence the node identity, including cancelled
                // work whose original worker process has disappeared.
                auto at = result.find(job.lease->worker_identity);
                if (at == result.end()) continue;
                if (other_owner_tasks && job.owner != observer) ++*other_owner_tasks;
                if (!fits(job.lease->task.resources, at->second)) at->second = Resources{0, 0, 0, 0};
                else subtract(at->second, job.lease->task.resources);
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
        if (task.compatibility != (task.kind == Kind::Pack ? offer.codec_compatibility : offer.compatibility))
            return reject(WaitReason::Incompatible);
        if (task.k < offer.min_k || task.k > offer.max_k) return reject(WaitReason::UnsupportedParameters);
        auto candidate = [&](Kind kind) -> std::optional<Candidate>
        {
            if (!(offer.capabilities & capability(kind))) return reject(WaitReason::UnsupportedKind);
            if (task.independent_members && (kind != Kind::GenerateGpu || !(offer.independent_k_mask & (1u << task.k)) ||
                (task.resources.cpu_threads ? task.resources.cpu_threads : offer.generation_threads) < 2))
                return reject(WaitReason::UnsupportedKind);
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
                    auto const extra = task.independent_members ? 0 : task.work_units - 1;
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
            *it = History{selected, task.k, task.strength, threads, owner, task.generation_output, task.independent_members}; return *it;
        }
        worker.histories.push_back(History{selected, task.k, task.strength, threads, owner, task.generation_output, task.independent_members});
        return worker.histories.back();
    }
    struct DatedCost
    {
        uint64_t low = UINT64_MAX, high = 0;
        unsigned samples = 0;
        void observe(Clock::time_point at, Clock::time_point now, uint64_t elapsed)
        {
            // Use the same one-day limit as the owner's construction calibration.
            if (at > now || now - at >= std::chrono::hours(24)) return;
            low = std::min(low, elapsed); high = std::max(high, elapsed); ++samples;
        }
    };
    struct DatedRoutes
    {
        DatedCost fastest;
        uint64_t other_low = UINT64_MAX;
        std::string const* worker = nullptr;
        Candidate route{};
        void observe(std::string const& id, Candidate candidate, DatedCost const& cost)
        {
            if (cost.samples >= 2 && (!worker || cost.high < fastest.high))
            {
                other_low = std::min(other_low, fastest.low);
                fastest = cost; worker = &id; route = candidate;
            }
            else other_low = std::min(other_low, cost.low);
        }
        bool separated() const { return worker && fastest.high < other_low; }
    };
    long double recent_cost(std::string const& identity, Worker const& worker, Task const& task, Kind kind, std::string const& owner,
                            Resources const& resources, Clock::time_point now, bool lower = false, DatedCost* dated = nullptr) const
    {
        auto const since = now - std::chrono::milliseconds(std::max(options.lease_ms, options.worker_ms));
        uint64_t result = 0;
        auto const& source = owners.at(owner);
        auto const path = std::find_if(source.paths.begin(), source.paths.end(), [&](auto const& item)
            { return item.first == identity; });
        for (auto const& history : worker.histories)
            if ((path != source.paths.end() || history.owner == owner) && history.matches(task, kind, history.owner, resources.cpu_threads))
                for (auto const& sample : std::span(history.samples).first(history.count))
                    if (sample.units == task.work_units && sample.resources == resources)
                    {
                        bool const bound = path != source.paths.end() && sample.origin == path->second;
                        if (sample.at > since && (bound || (history.owner == owner && (sample.origin.empty() || path == source.paths.end()))))
                            result = lower && result ? std::min(result, sample.observation.elapsed_ns) :
                                std::max(result, sample.observation.elapsed_ns);
                        else if (bound && dated) dated->observe(sample.at, now, sample.observation.elapsed_ns);
                    }
        return result ? result : std::numeric_limits<long double>::max();
    }
    struct DeliveryCost { long double ns = std::numeric_limits<long double>::max(); uint64_t assigned = 0; };
    static DeliveryCost delivery_cost(Worker const& worker, Task const& task, Kind kind, std::string const& owner, uint32_t threads)
    {
        DeliveryCost exact, prior;
        for (auto const& sample : worker.histories)
        {
            if (!sample.matches(task, kind, owner, threads, false)) continue;
            if (sample.independent == task.independent_members)
            {
                exact = {sample.cost(task.work_units), sample.assigned};
                if (sample.count) return exact;
            }
            // An ordinary contribution singleton and an independent stream
            // deliver the same chunks. Joint encoding has its own cost.
            else if (kind == Kind::GenerateGpu && (task.independent_members || task.work_units == 1))
                prior = {sample.cost(task.work_units, !sample.independent), sample.assigned};
        }
        if (prior.ns < exact.ns) return {prior.ns, std::max(prior.assigned, exact.assigned)};
        return exact;
    }
    using Nanoseconds = std::chrono::duration<long double, std::nano>;
    static long double pending_delivery_cost(std::vector<Job const*> const& deliveries, Worker const& worker,
                                             Job const& waiting, Candidate const& candidate, Clock::time_point now, bool& active)
    {
        auto const& task = waiting.requested;
        Job const* latest = nullptr;
        long double elapsed = 0;
        for (auto const* job : deliveries)
        {
            auto const& lease = *job->lease;
            auto const& completed = lease.task;
            if (job->owner != waiting.owner || job->granted_at < worker.route_since ||
                lease.worker_instance != worker.offer.instance || lease.endpoint != worker.offer.endpoint ||
                completed.kind != candidate.kind || completed.compatibility != task.compatibility ||
                completed.resources.cpu_threads != candidate.resources.cpu_threads || completed.k != task.k ||
                completed.strength != task.strength || completed.generation_output != task.generation_output) continue;
            // A pending singleton is also a lower bound on the time required
            // to deliver those same chunks in an independent stream.
            if (!(task.independent_members && completed.work_units == 1) &&
                (completed.independent_members != task.independent_members || completed.work_units != task.work_units)) continue;
            if (job->state == State::Finished)
            {
                if (!latest || job->unreported_finish > latest->unreported_finish) latest = job;
            }
            else { active = true; elapsed = std::max(elapsed, Nanoseconds(now - job->granted_at).count()); }
        }
        if (latest) return std::max(elapsed, Nanoseconds(latest->unreported_finish - latest->granted_at).count());
        return active ? elapsed : std::numeric_limits<long double>::max();
    }
    static long double remaining_on_lease(Worker const& worker, Job const& waiting, Job const& running, Clock::time_point now)
    {
        auto const& lease = *running.lease;
        auto const& held = lease.task.resources;
        auto const active_ns = delivery_cost(worker, lease.task, lease.task.kind, running.owner, held.cpu_threads).ns;
        auto const elapsed = Nanoseconds(now - running.granted_at).count();
        auto finish = active_ns - elapsed;
        if (finish <= 0)
        {
            // A shorter pipeline still pays setup for the whole lease. Allow
            // observed variation before spilling to a slower idle worker.
            auto const sample = std::find_if(worker.histories.begin(), worker.histories.end(), [&](auto const& item)
                { return item.matches(lease.task, lease.task.kind, running.owner, held.cpu_threads); });
            if (sample != worker.histories.end()) finish = sample->remaining_ns(lease.task.work_units, elapsed);
            if (finish <= 0 && sample != worker.histories.end() && !lease.task.independent_members && lease.task.work_units > 1)
                finish = sample->remaining_ns(lease.task.work_units, elapsed, true);
            if (finish <= 0 && lease.task.kind == Kind::GenerateGpu && lease.task.work_units > 1 &&
                (!lease.task.independent_members || (waiting.requested.independent_members && waiting.requested.work_units > 1)))
            {
                // A late batch can still be delivering its last member. Before
                // committing another whole pipeline, allow one measured
                // singleton on the running route to bound that wait.
                auto const ordinary = std::find_if(worker.histories.begin(), worker.histories.end(), [&](auto const& item)
                    { return !item.independent && item.matches(lease.task, lease.task.kind, running.owner, held.cpu_threads, false); });
                if (ordinary != worker.histories.end())
                {
                    auto const member = ordinary->cost(1, true);
                    if (member < std::numeric_limits<long double>::max()) finish = active_ns - elapsed + member;
                }
            }
            if (finish <= 0 && lease.task.kind == Kind::GenerateGpu && (lease.task.independent_members || lease.task.work_units == 1))
            {
                // A singleton delivers the same chunks as one member of an
                // independent pipeline. The next larger measured lease can
                // still bound its completion after its own samples run out.
                auto const pipeline = std::find_if(worker.histories.begin(), worker.histories.end(), [&](auto const& item)
                    { return item.independent && item.matches(lease.task, lease.task.kind, running.owner, held.cpu_threads, false); });
                if (pipeline != worker.histories.end()) finish = pipeline->remaining_ns(lease.task.work_units, elapsed, true);
            }
        }
        return finish;
    }
    bool earlier_on_release(std::vector<Job*> const& queue, Job const& waiting, std::vector<Job const*> const& active,
        Availability const& free, std::string const& idle, Kind idle_kind, long double idle_ns, Clock::time_point now, bool recent = false) const
    {
        struct Route
        {
            std::string const* identity;
            Worker const* worker;
            Resources held, capacity;
            Kind kind;
            long double ready;
        };
        std::vector<Route> routes;
        for (auto const* running : active)
        {
            auto const& lease = *running->lease;
            if (lease.worker_identity == idle && lease.task.kind == idle_kind) continue;
            auto const at = workers.find(lease.worker_identity);
            if (at == workers.end() || at->second.offer.instance != lease.worker_instance ||
                at->second.offer.endpoint != lease.endpoint || running->granted_at < at->second.route_since) continue;
            // The current owner's delivery predicts when the lease releases.
            // Each queued owner still needs its own estimate on this route.
            auto const& held = lease.task.resources;
            long double ready;
            if (recent)
            {
                auto const upper = recent_cost(at->first, at->second, lease.task, lease.task.kind, running->owner, held, now);
                if (upper == std::numeric_limits<long double>::max()) continue;
                ready = upper - Nanoseconds(now - running->granted_at).count();
            }
            else ready = remaining_on_lease(at->second, waiting, *running, now);
            if (ready <= 0 || ready >= idle_ns || ready >= Nanoseconds(running->expires - now).count()) continue;
            auto capacity = free.at(at->first);
            capacity.slots += held.slots; capacity.cpu_threads += held.cpu_threads;
            capacity.host_bytes += held.host_bytes; capacity.gpu_bytes += held.gpu_bytes;
            routes.push_back({&at->first, &at->second, held, capacity, lease.task.kind, ready});
        }
        // Predict the owner's ready queue on the routes that can release
        // capacity before this idle worker would deliver. Reuse each lease's
        // resources and use a worker's unreserved capacity only once.
        for (auto const* job : queue)
        {
            auto const deadline = Nanoseconds(job->start_by - now).count();
            auto best = routes.end();
            auto finish = idle_ns;
            Resources required;
            for (auto at = routes.begin(); at != routes.end(); ++at)
            {
                if (at->ready >= deadline || at->ready >= finish) continue;
                auto const next = eligible(*job, *at->identity, *at->worker, at->capacity, at->kind);
                if (!next) continue;
                auto const cost = recent ? recent_cost(*at->identity, *at->worker, job->requested, next->kind, job->owner, next->resources, now) :
                    delivery_cost(*at->worker, job->requested, next->kind, job->owner, next->resources.cpu_threads).ns;
                if (cost >= finish - at->ready) continue;
                best = at; finish = at->ready + cost; required = next->resources;
            }
            if (best == routes.end()) return false;
            best->ready = finish;
            if (!fits(required, best->held))
                for (auto& route : routes)
                    if (&route != &*best && route.worker == best->worker) route.capacity = route.held;
        }
        return true;
    }
    void schedule()
    {
        auto capacity = available();
        std::map<std::string, std::vector<Job*>> queued;
        std::vector<Job const*> active_generation;
        std::map<std::string, std::vector<Job const*>> pending_delivery;
        auto const recent = Clock::now() - std::chrono::milliseconds(options.lease_ms);
        for (auto& [key, job] : jobs)
            if (job.state == State::Queued && owners.contains(job.owner)) queued[job.owner].push_back(&job);
            else if ((job.state == State::Granted || job.state == State::Running) &&
                job.lease->task.kind != Kind::Pack && active_owner(job))
            {
                active_generation.push_back(&job);
                if (!job.observed) pending_delivery[job.lease->worker_identity].push_back(&job);
            }
            else if (job.state == State::Finished && job.unreported_finish != Clock::time_point{} &&
                job.unreported_finish > recent && active_owner(job)) pending_delivery[job.lease->worker_identity].push_back(&job);
        for (auto& [owner, queue] : queued)
            std::sort(queue.begin(), queue.end(), [](Job const* a, Job const* b)
            { return a->requested.sequence < b->requested.sequence; });
        // Weighted virtual service chooses among ready owners. Idle shares are
        // lent immediately; queued incompatible work does not block peers.
        for (;;)
        {
            advance_service_floor();
            std::vector<std::string> waiting;
            for (auto const& [key, queue] : queued) if (!queue.empty()) waiting.push_back(key);
            if (waiting.empty()) return;
            auto start = std::upper_bound(waiting.begin(), waiting.end(), last_owner);
            std::rotate(waiting.begin(), start, waiting.end());
            std::stable_sort(waiting.begin(), waiting.end(), [&](std::string const& a, std::string const& b)
            { return owners.at(a).service < owners.at(b).service; });
            bool granted = false;
            for (auto const& owner_id : waiting)
            {
                auto& queue = queued.at(owner_id);
                std::vector<std::string> candidates;
                for (auto const& [identity, worker] : workers) candidates.push_back(identity);
                auto worker_start = std::upper_bound(candidates.begin(), candidates.end(), last_worker);
                std::rotate(candidates.begin(), worker_start, candidates.end());
                std::string chosen;
                Kind chosen_kind = Kind::GenerateCpu;
                Resources chosen_resources;
                auto const now = Clock::now();
                // A wide request can wait for the busy GPU while a smaller
                // request from the same owner finishes on the idle GPU.
                auto job_it = std::find_if(queue.begin(), queue.end(), [&](Job const* job)
                {
                    chosen.clear();
                    long double best_cost = std::numeric_limits<long double>::max();
                    auto const other_work = Nanoseconds(job->other_work_until - now).count();
                    bool const gpu = job->requested.kind == Kind::GenerateGpu || job->requested.kind == Kind::GenerateAny;
                    uint64_t oldest_age = 0;
                    bool explore = false, awaiting_first = false, incumbent = false;
                    DatedRoutes dated_routes;
                    for (auto const& identity : candidates)
                    {
                        auto& worker = workers.at(identity);
                        for (auto const kind : {Kind::GenerateGpu, Kind::GenerateCpu, Kind::Pack})
                        {
                            auto selected = eligible(*job, identity, worker, capacity.at(identity), kind);
                            if (!selected) continue;
                            auto const sample = delivery_cost(worker, job->requested, selected->kind, job->owner, selected->resources.cpu_threads);
                            auto cost = sample.ns;
                            if (job->finite)
                            {
                                DatedCost dated;
                                auto const prior = recent_cost(identity, worker, job->requested, selected->kind, job->owner,
                                    selected->resources, now, false, pending_delivery.contains(identity) ? nullptr : &dated);
                                if (prior < std::numeric_limits<long double>::max())
                                {
                                    if (!incumbent || prior < best_cost)
                                    {
                                        incumbent = true; chosen = identity; chosen_kind = selected->kind;
                                        chosen_resources = selected->resources; best_cost = prior; explore = awaiting_first = false;
                                    }
                                    continue;
                                }
                                if (incumbent) continue;
                                dated_routes.observe(identity, *selected, dated);
                            }
                            uint64_t const age = assignments - sample.assigned;
                            bool const unmeasured = sample.ns == std::numeric_limits<long double>::max();
                            // A worker with spare slots can still be sending
                            // its first delivery. Finish can free the last
                            // slot before the owner accepts the last frame.
                            bool active_trial = false;
                            if (unmeasured && !pending_delivery.empty())
                            {
                                auto const pending = pending_delivery.find(identity);
                                if (pending != pending_delivery.end())
                                    cost = pending_delivery_cost(pending->second, worker, *job, *selected, now, active_trial);
                            }
                            bool const provisional = cost != sample.ns;
                            // Revisit measured GPU routes during enough
                            // parallel work to cover their delivery time.
                            bool const trial = !provisional && (!sample.assigned || (unmeasured && age >= 8) ||
                                (age >= 32 && (!gpu || cost <= other_work)));
                            if (chosen.empty() || (trial && (!explore || age > oldest_age)) ||
                                (!trial && !explore && std::pair(active_trial, cost) < std::pair(awaiting_first, best_cost)))
                            { chosen = identity; chosen_kind = selected->kind; chosen_resources = selected->resources;
                                best_cost = cost; oldest_age = age; explore = trial; awaiting_first = active_trial; }
                        }
                    }
                    if (chosen.empty()) return false;
                    // A finite group has no independent delivery to hide a new route's trial.
                    // A recent exact-width route can release in time for every ready request.
                    if (job->finite)
                    {
                        if (waiting.size() == 1 && earlier_on_release(queue, *job, active_generation, capacity, chosen, chosen_kind,
                            incumbent ? best_cost : std::numeric_limits<long double>::max(), now, true)) return false;
                        // Older receipts only compare immediately available workers.
                        // Overlapping intervals leave the ordinary trial order in place.
                        if (!incumbent && dated_routes.separated())
                        { chosen = *dated_routes.worker; chosen_kind = dated_routes.route.kind; chosen_resources = dated_routes.route.resources; }
                        return true;
                    }
                    // The owner can see planned fill work beyond the jobs
                    // that currently have receive buffers and a scratch grant.
                    if (!awaiting_first && best_cost <= other_work) return true;
                    // Elapsed time on an unfinished first task is a floor,
                    // not evidence that its other slots are quicker. Let a
                    // measured route serve a queue it can handle before the
                    // jobs' deadlines. Other owners retain normal admission.
                    if (awaiting_first && waiting.size() == 1)
                        return !earlier_on_release(queue, *job, active_generation, capacity, chosen, chosen_kind,
                            std::numeric_limits<long double>::max(), now);
                    return chosen_kind == Kind::Pack || explore || best_cost == std::numeric_limits<long double>::max() ||
                        waiting.size() != 1 || !earlier_on_release(queue, *job, active_generation, capacity, chosen, chosen_kind, best_cost, now);
                });
                if (job_it != queue.end())
                {
                    auto& job = **job_it;
                    auto const& identity = chosen;
                    auto& worker = workers.at(identity);
                    if (assignments == UINT64_MAX)
                    {
                        assignments = 0;
                        for (auto& [id, value] : workers) for (auto& sample : value.histories) sample.assigned = 0;
                    }
                    job.assignment = ++assignments;
                    history(worker, job.requested, chosen_kind, job.owner, chosen_resources.cpu_threads).assigned = job.assignment;
                    auto& owner = owners.at(owner_id);
                    Lease lease; lease.token = random_token(); lease.epoch = epoch; lease.owner_identity = owner.identity;
                    lease.broker_identity = broker_identity; lease.broker_node_id = broker_node_id;
                    lease.owner_id = owner.session.owner_id; lease.owner_session = owner.session.token;
                    lease.owner_generation = owner.session.generation; lease.worker_identity = identity;
                    lease.worker_instance = worker.offer.instance; lease.endpoint = worker.offer.endpoint;
                    lease.task = job.requested; lease.task.kind = chosen_kind; lease.task.resources = chosen_resources; lease.ttl_ms = options.lease_ms;
                    job.lease = std::move(lease); job.state = State::Granted;
                    subtract(capacity.at(identity), chosen_resources); queue.erase(job_it);
                    job.granted_at = job.touched = now;
                    job.expires = now + std::chrono::milliseconds(options.lease_ms);
                    if (chosen_kind != Kind::Pack)
                    { active_generation.push_back(&job); pending_delivery[identity].push_back(&job); }
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
    void check_profiles(WorkerOffer const& offer, std::vector<GenerationProfile> const& profiles, uint32_t threads = 0) const
    {
        require(profiles.size() <= max_generation_profiles, "too many generation profiles");
        for (auto it = profiles.begin(); it != profiles.end(); ++it)
        {
            auto const& value = *it;
            require((value.kind == Kind::GenerateCpu || value.kind == Kind::GenerateGpu) &&
                (offer.capabilities & capability(value.kind)) && value.k >= offer.min_k && value.k <= offer.max_k &&
                !(value.k & 1) && value.strength <= 32 && value.cpu_threads == (threads ? threads : offer.generation_threads) &&
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
        auto const threads = update->generation_profiles && !update->generation_profiles->empty() ?
            update->generation_profiles->front().cpu_threads : std::min(offer.generation_threads, update->capacity.cpu_threads);
        require(threads && threads <= update->capacity.cpu_threads,
            "capacity is below the worker generation CPU policy");
        if (update->generation_profiles) check_profiles(offer, *update->generation_profiles, threads);
        else check_profiles(offer, offer.generation_profiles, threads);
        check_capacity(lease.worker_identity, update->capacity, &releasing);
        offer.capacity = update->capacity; offer.capacity_revision = update->revision;
        if (offer.capacity.gpu_bytes) offer.gpu_recovery.clear();
        offer.generation_threads = threads;
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
    static WaitReason reserved_reason(WaitReason reason)
    {
        switch (reason)
        {
        case WaitReason::CpuCapacity: return WaitReason::CpuReserved;
        case WaitReason::HostCapacity: return WaitReason::HostReserved;
        case WaitReason::GpuCapacity: return WaitReason::GpuReserved;
        case WaitReason::SlotCapacity: return WaitReason::SlotReserved;
        default: return WaitReason::ResourcesReserved;
        }
    }
    WaitReason waiting(Job const& job, Availability const& capacity) const
    {
        if (job.state == State::Cancelled) return WaitReason::OwnerCancelled;
        if (job.state != State::Queued) return WaitReason::None;
        WaitReason reason = WaitReason::NoWorkers, busy = WaitReason::None;
        for (auto const& [identity, worker] : workers)
        {
            WaitReason candidate = WaitReason::None;
            if (eligible(job, identity, worker, worker.offer.capacity, {}, &candidate))
            {
                if (eligible(job, identity, worker, capacity.at(identity), {}, &candidate)) return WaitReason::FairQueue;
                busy = reserved_reason(candidate);
                continue;
            }
            if (uint8_t(candidate) > uint8_t(reason)) reason = candidate;
        }
        return busy != WaitReason::None ? busy : reason;
    }
    void status(Reader& reader, Writer& response, bool recovery)
    {
        auto const section = StatusSection(reader.u8()); auto const cursor = reader.u64(); auto const limit = reader.u32(); reader.finish();
        require(section >= StatusSection::Workers && section <= StatusSection::Jobs && limit && limit <= max_status_page_size,
            "invalid broker status page request");
        uint64_t queued = 0, granted = 0, running = 0, cancelled = 0;
        auto const capacity = section == StatusSection::Owners ? Availability{} : available();
        std::map<std::string, std::array<uint64_t, 4>> owner_counts;
        if (section == StatusSection::Owners)
            for (auto const& [key, owner] : owners) owner_counts.emplace(key, std::array<uint64_t, 4>{});
        for (auto const& [key, job] : jobs)
        {
            queued += job.state == State::Queued; granted += job.state == State::Granted;
            running += job.state == State::Running; cancelled += job.state == State::Cancelled;
            if (auto at = owner_counts.find(job.owner); at != owner_counts.end())
            {
                auto& counts = at->second;
                counts[0] += job.state == State::Queued; counts[1] += job.state == State::Granted;
                counts[2] += job.state == State::Running; counts[3] += job.state == State::Cancelled;
            }
        }
        auto const total = section == StatusSection::Workers ? workers.size() : section == StatusSection::Owners ? owners.size() : jobs.size();
        auto const count = cursor >= total ? 0 : std::min<uint64_t>(limit, total - cursor);
        response.u8(uint8_t(section)); response.string(profile_detail::host_name());
        response.u64(cursor < total && count < total - cursor ? cursor + count : 0);
        response.u64(workers.size()); response.u64(owners.size()); response.u64(jobs.size());
        response.u64(queued); response.u64(granted); response.u64(running); response.u64(cancelled); response.u32(uint32_t(count));
        uint64_t index = 0;
        auto selected = [&] { return index >= cursor && index - cursor < count; };
        if (section == StatusSection::Workers) for (auto const& [identity, worker] : workers)
        {
            if (selected())
            {
                auto reserved_capacity = worker.offer.capacity; subtract(reserved_capacity, capacity.at(identity));
                response.string(identity); response.string(worker.offer.instance); response.string(worker.offer.compatibility);
                response.string(worker.offer.codec_compatibility); response.string(worker.offer.producer_build);
                response.u32(worker.offer.capabilities);
                write_endpoint(response, worker.offer.endpoint);
                write_resources(response, worker.offer.capacity); write_resources(response, reserved_capacity);
                response.u8(uint8_t(worker.offer.admission));
                response.u64(uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - worker.seen).count()));
                write_worker_hardware(response, worker.offer.hardware);
                if (recovery) response.string(worker.offer.gpu_recovery);
            }
            ++index; if (index >= cursor && index - cursor >= count) break;
        }
        else if (section == StatusSection::Owners) for (auto const& [key, owner] : owners)
        {
            if (selected())
            {
                auto const& counts = owner_counts.at(key);
                response.string(owner.identity); response.string(owner.session.owner_id); response.u64(counts[0]);
                response.u64(counts[1]); response.u64(counts[2]); response.u64(counts[3]);
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
                response.u8(uint8_t(task.kind)); response.u8(uint8_t(job.state)); response.u8(uint8_t(waiting(job, capacity)));
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
        case Operation::Status:
        case Operation::StatusGpuRecovery: status(reader, response, operation == Operation::StatusGpuRecovery); break;
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
        case Operation::AcquireFiniteGeneration:
        {
            auto session = read_session(reader); auto task = read_task(reader);
            auto const remaining = reader.u32(), ahead = reader.u32(); reader.finish();
            require(!ahead || (task.kind != Kind::Pack && ahead <= remaining), "invalid generation backlog");
            bool const finite = operation == Operation::AcquireFiniteGeneration;
            require(!finite || (!ahead && task.kind != Kind::Pack && task.generation_output != GenerationOutput::Contributions),
                "invalid finite generation request");
            auto const now = Clock::now();
            auto const start_by = remaining ? now + std::chrono::milliseconds(remaining) : Clock::time_point::max();
            auto const other_work_until = ahead ? now + std::chrono::milliseconds(ahead) : Clock::time_point{};
            auto& source = owner(identity, session);
            require(identifier(task.task_id, 256) && identifier(task.compatibility, 256) && task.attempt && task.sequence && task.work_units &&
                task.k >= 18 && task.k <= 28 && !(task.k & 1) && task.strength <= 32 && task.resources.slots &&
                task.resources.host_bytes && (task.kind == Kind::GenerateCpu || task.kind == Kind::GenerateGpu ||
                    task.kind == Kind::GenerateAny || task.kind == Kind::Pack) &&
                (task.kind != Kind::Pack || task.resources.cpu_threads == 1) &&
                (task.kind == Kind::Pack || task.work_units <= max_generation_batch_members), "invalid compute task");
            require((task.kind != Kind::GenerateCpu && task.kind != Kind::Pack) || !task.resources.gpu_bytes,
                    "CPU task cannot reserve GPU memory");
            require(!task.independent_members || task.kind == Kind::GenerateGpu, "independent members require GPU generation");
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
                    job.requested = task; job.lease.reset(); job.state = State::Queued;
                    job.observed = false; job.unreported_finish = {};
                    job.start_by = Clock::time_point::max();
                }
                else require(task == job.requested, "stale or changed task attempt");
                require(job.state == State::Queued || job.state == State::Granted, "task needs a new attempt");
                job.touched = Clock::now();
            }
            auto& job = it->second; job.start_by = std::min(job.start_by, start_by);
            job.other_work_until = other_work_until; job.finite = finite; schedule();
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
                if (job.state == State::Granted) release_unstarted(job);
                else if (job.state == State::Running) job.state = State::Cancelled;
                else if (job.state == State::Queued) job.state = State::Finished;
                job.unreported_finish = {}; job.touched = Clock::now();
            }
            schedule(); break;
        }
        case Operation::Register:
        case Operation::RegisterWorkerRoute:
        case Operation::RegisterGpuRecovery:
        {
            auto offer = read_offer(reader);
            bool retain_route = operation == Operation::RegisterWorkerRoute;
            if (operation == Operation::RegisterGpuRecovery)
            {
                auto const flag = reader.u8(); offer.gpu_recovery = gpu_recovery_reason(reader.string(240));
                require(flag <= 1 && !offer.capacity.gpu_bytes && !offer.gpu_recovery.empty() &&
                    (offer.capabilities & capability(Kind::GenerateGpu)), "invalid withdrawn GPU registration");
                retain_route = flag;
            }
            if (offer.independent_k_mask)
            {
                uint32_t supported = 0;
                for (auto const& profile : offer.generation_profiles)
                    if (profile.kind == Kind::GenerateGpu && profile.k <= 28) supported |= 1u << profile.k;
                require(!(offer.independent_k_mask & ~supported) &&
                    (offer.capabilities & capability(Kind::GenerateGpu)), "independent members require a GPU profile");
            }
            reader.finish();
            require(identifier(offer.instance, 64) && identifier(offer.compatibility, 256) &&
                (!(offer.capabilities & capability(Kind::Pack)) || identifier(offer.codec_compatibility, 256)) &&
                offer.producer_build.size() <= 128 && !offer.endpoint.host.empty() &&
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
            if (!offer.hardware.host.empty())
                require(offer.hardware.logical_cpus && offer.hardware.partition.cpu_threads >= offer.capacity.cpu_threads &&
                    offer.hardware.partition.slots >= offer.capacity.slots &&
                    offer.hardware.partition.host_bytes >= offer.capacity.host_bytes &&
                    offer.hardware.partition.gpu_bytes >= offer.capacity.gpu_bytes &&
                    offer.hardware.standing_host_bytes < offer.hardware.partition.host_bytes &&
                    offer.hardware.observed_at_unix_ms, "invalid worker hardware observation");
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
                if (retain_route) offer.endpoint = saved.endpoint;
                require(saved.compatibility == offer.compatibility && saved.codec_compatibility == offer.codec_compatibility &&
                    saved.capabilities == offer.capabilities && saved.independent_k_mask == offer.independent_k_mask &&
                    saved.min_k == offer.min_k && saved.max_k == offer.max_k && saved.codec_limits == offer.codec_limits,
                    "worker registration changed without a new instance");
                require(saved.hardware.host == offer.hardware.host && saved.hardware.gpu_identity == offer.hardware.gpu_identity,
                    "worker physical binding changed without a new instance");
                if (offer.capacity_revision > saved.capacity_revision)
                {
                    check_capacity(identity, offer.capacity); saved.capacity = offer.capacity;
                    saved.capacity_revision = offer.capacity_revision; saved.generation_profiles = offer.generation_profiles;
                    saved.generation_threads = offer.generation_threads;
                    // Existing leases retain their original route and identity.
                    // New grants start fresh delivery history on the new route.
                    if (saved.endpoint != offer.endpoint)
                    { it->second.histories.clear(); it->second.route_since = Clock::now(); }
                    saved.endpoint = offer.endpoint;
                }
                else if (offer.capacity_revision == saved.capacity_revision)
                    require(saved.capacity == offer.capacity && saved.generation_profiles == offer.generation_profiles &&
                        saved.generation_threads == offer.generation_threads &&
                        saved.endpoint == offer.endpoint, "capacity revision changed its value");
                if (saved.capacity_revision == offer.capacity_revision) saved.gpu_recovery = std::move(offer.gpu_recovery);
                saved.admission = offer.admission; saved.producer_build = offer.producer_build;
                saved.hardware = std::move(offer.hardware); it->second.seen = Clock::now();
            }
            response.string(epoch);
            if (operation != Operation::Register) write_endpoint(response, workers.at(identity).offer.endpoint);
            schedule(); break;
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
            auto result = BeginResult::Stale;
            if (job.state == State::Granted && active_owner(job) && worker != workers.end() &&
                worker->second.offer.instance == lease.worker_instance)
            {
                if (worker->second.offer.admission != WorkerAdmission::Accepting) result = BeginResult::AwaitingAdmission;
                else
                {
                    result = BeginResult::Running; job.state = State::Running;
                    job.expires = Clock::now() + std::chrono::milliseconds(options.lease_ms);
                }
            }
            response.u8(uint8_t(result)); break;
        }
        case Operation::Heartbeat:
        {
            auto instance = reader.string(64); auto revision = reader.u64(); auto const admission = reader.u8(); reader.finish();
            require(identifier(instance, 64) && revision && admission <= uint8_t(WorkerAdmission::Drained), "invalid worker heartbeat");
            auto worker = workers.find(identity);
            bool const current = worker != workers.end() && worker->second.offer.instance == instance &&
                worker->second.offer.capacity_revision == revision;
            if (current)
            {
                worker->second.seen = Clock::now();
                if (admission && WorkerAdmission(admission) != worker->second.offer.admission)
                { worker->second.offer.admission = WorkerAdmission(admission); schedule(); }
            }
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
            if (job.state != State::Finished)
            {
                update_capacity(lease, update, job);
                if (job.state == State::Running && !job.observed && lease.task.kind != Kind::Pack)
                    job.unreported_finish = Clock::now();
            }
            job.state = State::Finished; job.touched = Clock::now(); schedule(); break;
        }
        case Operation::Reject:
        {
            auto lease = read_lease(reader); auto reason = FailureCode(reader.u8()); auto update = read_update(reader);
            reader.finish();
            require(identity == lease.worker_identity && (reason == FailureCode::UnsupportedResources ||
                (reason == FailureCode::RefusedBeforeBegin && !update)), "invalid worker admission rejection");
            auto& job = lease_job(lease);
            if (reason == FailureCode::RefusedBeforeBegin)
            {
                if (job.state != State::Granted || !active_owner(job)) throw StaleLease("worker refusal needs a current unstarted grant");
                release_unstarted(job);
                // The worker attested that it never began this exact lease.
                // An owner whose upload connection closed cannot abandon it later.
                job.lease.reset(); job.touched = Clock::now(); schedule(); break;
            }
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
            job.unreported_finish = {}; job.touched = Clock::now();
            bool const known = std::any_of(job.failures.begin(), job.failures.end(), [&](Job::Failure const& failure)
            { return failure.worker == lease.worker_identity && failure.instance == lease.worker_instance && failure.kind == Kind::GenerateAny; });
            if (!known && job.failures.size() < max_retry_failures)
                job.failures.push_back({lease.worker_identity, lease.worker_instance, Kind::GenerateAny});
            schedule(); break;
        }
        case Operation::Delivered:
        case Operation::DeliveredOnPath:
        {
            auto lease = read_lease(reader); auto elapsed = reader.u64(); auto input = reader.u64(); auto output = reader.u64();
            std::string origin;
            if (operation == Operation::DeliveredOnPath)
            {
                auto path = reader.string(64);
                require(path.size() == 64 && path.find_first_not_of("0123456789abcdef") == path.npos, "invalid generation path");
                origin = identity + ':' + path;
            }
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
                if (worker->second.offer.endpoint == lease.endpoint && job.granted_at >= worker->second.route_since)
                    history(worker->second, lease.task, lease.task.kind, job.owner, lease.task.resources.cpu_threads).add({elapsed, input, output}, lease.task.work_units, lease.task.resources, std::move(origin));
                job.observed = true; job.unreported_finish = {}; break;
            }
            catch (StaleLease const& error) { throw StaleDelivery(error.what()); }
        }
        case Operation::GenerationPaths:
        {
            auto const session = read_session(reader); auto& source = owner(identity, session);
            auto const count = reader.u8(); require(count <= max_status_page_size, "excessive generation paths");
            BrokerClient::GenerationPaths paths; paths.reserve(count);
            for (unsigned i = 0; i < count; ++i)
            {
                auto worker = reader.string(64), path = reader.string(64);
                require((paths.empty() || paths.back().first < worker) && identifier(worker, 64) && path.size() == 64 && path.find_first_not_of("0123456789abcdef") == path.npos,
                    "invalid generation path");
                paths.emplace_back(std::move(worker), identity + ':' + path);
            }
            reader.finish(); source.paths = std::move(paths); break;
        }
        case Operation::FinitePackedRoute:
        {
            auto session = read_session(reader); Job job; job.requested = read_task(reader); reader.finish();
            (void)owner(identity, session); job.owner = owner_key(identity, session.owner_id);
            auto& t = job.requested;
            require(t.kind != Kind::Pack && t.generation_output == GenerationOutput::Packed && !t.independent_members &&
                t.work_units > 1 && t.work_units <= max_generation_batch_members, "invalid finite packed width");
            auto const room = available(); auto const now = Clock::now();
            std::pair<std::string, Kind> best; long double finish = std::numeric_limits<long double>::max();
            DatedRoutes dated_routes;
            for (auto const& [id, worker] : workers)
                for (auto kind : {Kind::GenerateGpu, Kind::GenerateCpu})
                    if (auto ready = eligible(job, id, worker, room.at(id), kind))
                    {
                        DatedCost dated;
                        auto const cost = recent_cost(id, worker, t, kind, job.owner, ready->resources, now, false,
                            room.at(id) == worker.offer.capacity ? &dated : nullptr);
                        if (cost < finish) { finish = cost; best = {id, kind}; }
                        dated_routes.observe(id, *ready, dated);
                    }
            bool const dated_batch = best.first.empty() && dated_routes.separated();
            if (dated_batch)
            {
                best = {*dated_routes.worker, dated_routes.route.kind};
                finish = dated_routes.fastest.high;
            }
            auto split_cost = [&](std::string const& id, Worker const& worker, Candidate const& next, bool lower = false)
            {
                DatedCost dated;
                auto const cost = recent_cost(id, worker, t, next.kind, job.owner, next.resources, now, lower,
                    dated_batch ? &dated : nullptr);
                // An older full-batch upper bound must beat the possible split.
                return cost == std::numeric_limits<long double>::max() && dated.samples ? static_cast<long double>(dated.low) : cost;
            };
            if (!best.first.empty())
            {
                // A worker can offer several slots under one identity. Compare
                // exact smaller leases in waves that fit its current resources.
                auto const width = uint32_t(t.work_units);
                auto const& worker = workers.at(best.first); auto const& free = room.at(best.first);
                std::array<long double, max_generation_batch_members + 1> split;
                split.fill(finish); split[0] = 0;
                for (uint32_t members = 1; members < width; ++members)
                {
                    t.work_units = members;
                    for (auto kind : {Kind::GenerateGpu, Kind::GenerateCpu})
                        if (auto next = eligible(job, best.first, worker, free, kind))
                        {
                            auto const cost = split_cost(best.first, worker, *next);
                            if (cost >= finish) continue;
                            auto const lanes = free.parallel_tasks(next->resources);
                            for (uint32_t filled = members; filled <= width; ++filled)
                                for (uint32_t count = 1; count <= std::min(lanes, filled / members); ++count)
                                    split[filled] = std::min(split[filled], split[filled - count * members] + cost);
                        }
                }
                if (split[width] < finish) best.first.clear();
            }
            // If another route can deliver an actual singleton no later than the
            // whole batch, keep those IDs available for ordinary split placement.
            if (!best.first.empty())
            {
                t.work_units = 1;
                for (auto const& [id, worker] : workers)
                    for (auto kind : {Kind::GenerateGpu, Kind::GenerateCpu})
                        if (id != best.first)
                            if (auto next = eligible(job, id, worker, dated_batch ? room.at(id) : worker.offer.capacity, kind))
                                if (split_cost(id, worker, *next, true) <= finish)
                                    best.first.clear();
            }
            response.string(best.first); if (!best.first.empty()) response.u8(uint8_t(best.second)); break;
        }
        case Operation::GenerationHint:
        case Operation::GenerationAvailability:
        case Operation::GenerationRoutes:
        case Operation::GenerationRoutesGpuRecovery:
        {
            auto session = read_session(reader); auto task = read_task(reader);
            bool const detailed = operation != Operation::GenerationHint;
            auto const codec = detailed ? reader.string(256) : std::string{};
            reader.finish(); (void)owner(identity, session);
            require(task.kind == Kind::GenerateCpu || task.kind == Kind::GenerateGpu || task.kind == Kind::GenerateAny,
                "generation hint requires a generation task");
            bool const independent_request = task.independent_members;
            auto const requested_output = task.generation_output;
            Job job; job.requested = std::move(task); uint32_t maximum = 0, slots = 0;
            uint32_t independent = independent_request ? 0 : max_generation_batch_members;
            std::array<uint32_t, max_generation_batch_members + 1> independent_slots{};
            GenerationRoster roster;
            bool const recovery = operation == Operation::GenerationRoutesGpuRecovery;
            bool const report_routes = operation == Operation::GenerationRoutes || recovery;
            GenerationAdmission admission; admission.registered = uint32_t(workers.size());
            bool codec_available = false, concurrent_codec = false;
            auto const free = report_routes ? available(owner_key(identity, session.owner_id), &admission.other_owner_tasks) :
                detailed ? available() : Availability{};
            for (auto const& [identity, worker] : workers)
            {
                auto const& capacity = worker.offer.capacity;
                if (recovery && !worker.offer.gpu_recovery.empty() && roster.recovering_gpus.size() < max_status_page_size)
                    roster.recovering_gpus.push_back({worker.offer.hardware.host.empty() ? identity : worker.offer.hardware.host,
                        worker.offer.gpu_recovery});
                job.requested.independent_members = false;
                job.requested.generation_output = requested_output;
                bool const codec_room = detailed && worker.offer.admission == WorkerAdmission::Accepting && worker.offer.codec_compatibility == codec &&
                    (worker.offer.capabilities & capability(Kind::Pack)) && free.at(identity).slots &&
                    free.at(identity).cpu_threads && free.at(identity).host_bytes;
                job.requested.work_units = 1;
                WaitReason reason = WaitReason::None;
                auto candidate = eligible(job, identity, worker, capacity, {}, report_routes ? &reason : nullptr);
                if (codec_room)
                {
                    codec_available = true;
                    concurrent_codec |= !candidate || (capacity.slots > candidate->resources.slots &&
                        capacity.cpu_threads > candidate->resources.cpu_threads && capacity.host_bytes > candidate->resources.host_bytes);
                }
                if (!candidate)
                {
                    if (report_routes)
                    {
                        if (worker.offer.admission == WorkerAdmission::Paused) reason = WaitReason::WorkerPaused;
                        else if (worker.offer.admission != WorkerAdmission::Accepting) reason = WaitReason::WorkerDraining;
                        ++admission.waiting[size_t(reason)];
                    }
                    continue;
                }
                auto const parallel = capacity.parallel_tasks(candidate->resources);
                slots += std::min(UINT32_MAX - slots, parallel);
                if (report_routes)
                {
                    ++admission.eligible;
                    auto const& remaining = free.at(identity);
                    auto const ready = remaining == capacity ? candidate : eligible(job, identity, worker, remaining, {}, &reason);
                    if (ready)
                        admission.available_slots += std::min({parallel, remaining.parallel_tasks(ready->resources),
                            UINT32_MAX - admission.available_slots});
                    else ++admission.waiting[size_t(reserved_reason(reason))];
                    if (roster.routes.size() == max_status_page_size) roster.complete = false;
                    else
                    {
                        auto lanes = [&](Kind kind)
                        {
                            if (candidate->kind == kind) return parallel;
                            auto const other = eligible(job, identity, worker, capacity, kind);
                            return other ? capacity.parallel_tasks(other->resources) : 0u;
                        };
                        auto const& offer = worker.offer;
                        roster.routes.push_back({identity, offer.instance, offer.endpoint, offer.hardware.host,
                            offer.hardware.gpu_identity, lanes(Kind::GenerateCpu), lanes(Kind::GenerateGpu)});
                    }
                }
                for (uint32_t count = max_generation_batch_members; count > maximum; --count)
                {
                    job.requested.work_units = count;
                    if (eligible(job, identity, worker, capacity)) { maximum = count; break; }
                }
                if (!detailed) continue;
                uint32_t width = 0;
                if (job.requested.kind != Kind::GenerateCpu && worker.offer.independent_k_mask)
                {
                    job.requested.independent_members = true;
                    job.requested.generation_output = GenerationOutput::Contributions;
                    for (uint32_t count = max_generation_batch_members; count; --count)
                    {
                        job.requested.work_units = count;
                        if (eligible(job, identity, worker, capacity, Kind::GenerateGpu)) { width = count; break; }
                    }
                }
                // Planning shares work over all eligible lanes. A formed
                // independent request needs only one matching worker.
                independent = independent_request ? std::max(independent, width) : std::min(independent, width);
                // Independent batches reuse the ordinary GPU singleton's resources.
                for (uint32_t n = 1; n <= width; ++n)
                    independent_slots[n] += std::min(UINT32_MAX - independent_slots[n], parallel);
            }
            response.u32(maximum); response.u32(slots);
            if (detailed)
            {
                response.u32(slots ? independent : 0);
                response.u8(uint8_t(codec_available) | (uint8_t(concurrent_codec) << 1));
                for (auto n : std::span(independent_slots).subspan(1)) response.u32(n);
            }
            if (report_routes)
            {
                response.u8(roster.complete); response.u8(uint8_t(roster.routes.size()));
                for (auto const& route : roster.routes)
                {
                    response.string(route.identity); response.string(route.instance); write_endpoint(response, route.endpoint);
                    response.string(route.host); response.string(route.gpu_identity);
                    response.u32(route.cpu_slots); response.u32(route.gpu_slots);
                }
                response.u32(admission.registered); response.u32(admission.eligible);
                response.u32(admission.available_slots); response.u32(admission.other_owner_tasks);
                for (auto number : admission.waiting) response.u32(number);
                if (recovery)
                {
                    response.u8(uint8_t(roster.recovering_gpus.size()));
                    for (auto const& gpu : roster.recovering_gpus) { response.string(gpu.host); response.string(gpu.cause); }
                }
            }
            break;
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
