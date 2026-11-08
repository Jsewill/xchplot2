// Broker control plane. Bulk results travel directly from worker to owner.
#pragma once
#include "host/ComputeNetwork.hpp"
#include <atomic>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>

namespace group_grow::pool
{
inline constexpr uint32_t protocol_version = 5;
inline constexpr size_t max_retry_failures = 16;
inline constexpr uint32_t max_generation_batch_members = 8;
inline constexpr size_t max_generation_profiles = 512;
inline constexpr unsigned control_channel_limit = 4;
inline constexpr uint32_t default_max_codec_chunks = 1024;
inline constexpr uint64_t default_codec_task_bytes = uint64_t(64) << 20;
// A serialized codec window begins with its count. Each chunk then carries
// ordinal, old membership/blob length and contribution/new-member counts.
inline constexpr uint32_t codec_window_header_bytes = 4, codec_chunk_header_bytes = 18;
enum class FailureCode : uint8_t { UnsupportedResources = 1 };
class UnsupportedResources : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
class StaleOwnerSession : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
class StaleDelivery : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
class StaleLease : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
enum class Kind : uint8_t { GenerateCpu = 1, GenerateGpu = 2, Pack = 3, GenerateAny = 4 };
inline constexpr uint32_t capability(Kind kind) { return 1u << uint8_t(kind); }
struct Resources
{
    // Generation requests may use zero for the selected worker's configured
    // generation policy. Concrete grants always carry positive CPU counts.
    uint32_t cpu_threads = 0, slots = 1;
    // Offers advertise active leasable budgets after fixed service overhead.
    // CPU/codec requests contain their conservative host bound. Generation
    // profiles refine the selected worker's exact conservative task demand.
    uint64_t host_bytes = 0, gpu_bytes = 0;
    bool operator==(Resources const&) const = default;
};
struct GenerationProfile
{
    Kind kind = Kind::GenerateGpu;
    uint8_t k = 18, strength = 0;
    uint32_t cpu_threads = 1;
    // Single member's dynamic peak, excluding retained service allocations
    // already removed from offer.capacity. Smaller thread grants retain this
    // conservative host allowance. Extra members add only their owned results.
    Resources resources;
    uint64_t member_host_bytes = 0;
    uint32_t max_batch_members = 1;
    bool operator==(GenerationProfile const&) const = default;
};
struct CodecLimits
{
    uint32_t max_chunks = default_max_codec_chunks;
    uint64_t max_task_bytes = default_codec_task_bytes;
    bool operator==(CodecLimits const&) const = default;
};
enum class WorkerAdmission : uint8_t { Accepting = 1, Paused, Draining, Drained };
struct WorkerOffer
{
    std::string instance, compatibility;
    // Generation and codec contracts are negotiated independently. The
    // producer's source identity is retained as provenance, never eligibility.
    std::string codec_compatibility, producer_build;
    Endpoint endpoint;
    Resources capacity;
    uint64_t capacity_revision = 1;
    uint32_t generation_threads = 1, capabilities = 0;
    uint8_t min_k = 18, max_k = 28;
    std::vector<GenerationProfile> generation_profiles;
    CodecLimits codec_limits;
    WorkerAdmission admission = WorkerAdmission::Accepting;
};
struct Task
{
    std::string task_id, compatibility;
    uint64_t attempt = 1, sequence = 1;
    // Generation members or exact serialized codec input bytes. Normalizes
    // observed delivery cost and checks each codec worker's input bound.
    uint64_t work_units = 1;
    Kind kind = Kind::GenerateCpu;
    uint8_t k = 18, strength = 0;
    Resources resources;
    // Pack's exact serialized input chunk count and requested receive bound.
    // Generation tasks leave both zero. Retries preserve these fields.
    uint32_t chunk_count = 0;
    uint64_t max_output_bytes = 0;
    bool operator==(Task const&) const = default;
};
struct OwnerSession
{
    std::string owner_id, token, epoch;
    uint64_t generation = 0;
};
struct Lease
{
    std::string token, epoch, owner_identity, owner_id, owner_session;
    // Authority is pinned to the authenticated TLS leaf, independently of its
    // restart epoch. A different enrolled broker cannot assert this identity.
    std::string broker_identity, broker_node_id;
    uint64_t owner_generation = 0;
    std::string worker_identity, worker_instance;
    Endpoint endpoint;
    // Concrete backend/resources may differ from the requested GenerateAny.
    Task task;
    unsigned ttl_ms = 0;
};
struct CapacityUpdate
{
    Resources capacity;
    uint64_t revision = 0;
    std::optional<std::vector<GenerationProfile>> generation_profiles = {};
};
struct DeliveryObservation
{
    uint64_t elapsed_ns = 0, input_bytes = 0, output_bytes = 0;
};
void write_lease(Writer&, Lease const&);
Lease read_lease(Reader&);

struct ControlMetrics
{
    uint64_t requests = 0, connections = 0, reused = 0;
    uint64_t request_bytes = 0, response_bytes = 0, wait_ns = 0;
};
inline constexpr uint32_t max_status_page_size = 32;
enum class StatusSection : uint8_t { Workers = 1, Owners, Jobs };
enum class JobState : uint8_t { Queued = 1, Granted, Running, Cancelled, Finished, Expired };
enum class WaitReason : uint8_t
{
    None, NoWorkers, Incompatible, UnsupportedKind, UnsupportedParameters,
    RetryExcluded, CpuCapacity, HostCapacity, GpuCapacity, SlotCapacity,
    WorkerPaused, WorkerDraining, ResourcesReserved, FairQueue, OwnerCancelled,
    CpuReserved, HostReserved, GpuReserved, SlotReserved
};
char const* admission_name(WorkerAdmission);
char const* job_state_name(JobState);
char const* wait_reason_name(WaitReason);
struct StatusRequest
{
    StatusSection section = StatusSection::Workers;
    uint64_t cursor = 0;
    uint32_t limit = max_status_page_size;
};
struct BrokerWorkerStatus
{
    std::string identity, instance, compatibility;
    std::string codec_compatibility, producer_build;
    uint32_t capabilities = 0;
    Endpoint endpoint;
    Resources capacity, reserved{0, 0, 0, 0};
    WorkerAdmission admission = WorkerAdmission::Accepting;
    uint64_t seen_ms = 0;
};
struct BrokerOwnerStatus
{
    std::string identity, owner_id;
    uint64_t queued = 0, granted = 0, running = 0, cancelled = 0;
};
struct BrokerJobStatus
{
    // Broker-local diagnostic sequence, never a session or lease token.
    uint64_t id = 0;
    std::string owner_identity, owner_id, worker_identity, worker_instance;
    Kind kind = Kind::GenerateCpu;
    JobState state = JobState::Queued;
    WaitReason waiting = WaitReason::None;
    Resources resources;
    uint64_t attempt = 0, work_units = 0, age_ms = 0;
    uint8_t k = 0, strength = 0;
};
struct BrokerStatusPage
{
    // Live pages may change between calls. next_cursor == 0 ends this section.
    StatusSection section = StatusSection::Workers;
    uint64_t next_cursor = 0, workers = 0, owners = 0, jobs = 0;
    uint64_t queued = 0, granted = 0, running = 0, cancelled = 0;
    std::vector<BrokerWorkerStatus> worker_records;
    std::vector<BrokerOwnerStatus> owner_records;
    std::vector<BrokerJobStatus> job_records;
};
// Bounded persistent TLS channels, shared by copies and per-task deadlines.
// A channel has one request in flight. A failed request is never replayed.
// Control frames use a fixed 65536-byte bound independent of bulk-data frames.
// Calls throw on authentication, protocol, broker restart, stale session, or
// stale attempt.
// acquire() returns empty while queued or a previous attempt still holds a
// reservation; callers wait with cancellation checks. New tasks submit strictly
// increasing sequences. Retry keeps task_id/sequence/body and raises attempt.
// Queued tasks expire after a lease interval without a matching acquire poll;
// a later retry needs a newer attempt, as for an expired grant.
class BrokerClient
{
public:
    BrokerClient(Endpoint endpoint, TlsOptions tls, unsigned channels = control_channel_limit);
    // Independent per-task copy; concurrent callers keep the shared base
    // client unchanged. A nested bound can only shorten its parent deadline.
    BrokerClient with_deadline(std::chrono::steady_clock::time_point) const;
    // Route changes are shared by task copies. The caller first authenticates
    // the alternate as the same broker epoch; requests are never replayed.
    void reroute(Endpoint const&);
    // Executor registry entries retain transport reservations while task
    // copies still share their channels, including during broker outages.
    bool shared() const noexcept;
    void disconnect_idle();
    OwnerSession open_owner(std::string const& owner_id, bool ephemeral = false);
    void close_owner(OwnerSession const&);
    std::optional<Lease> acquire(OwnerSession const&, Task const&);
    // Largest supported batch against each offer's total dynamic capacity,
    // independent of active leases. Advisory only; acquire still queues fairly.
    uint32_t generation_hint(OwnerSession const&, Task const&);
    void cancel(OwnerSession const&, std::string const& task_id, uint64_t attempt);
    // Only the worker renews Running work; unused grants have a fixed TTL.
    void renew(Lease const&);
    // The authenticated broker epoch selects the correct authority when one
    // executor accepts work from several independent brokers.
    std::string register_worker(WorkerOffer const&, std::string* peer_identity = nullptr, std::string* peer_node_id = nullptr);
    void unregister_worker(std::string const& instance);
    // Graceful rotation never releases granted or accepted work. False keeps
    // this authority and its transport reservation until the leases finish.
    bool retire_worker(std::string const& instance);
    // Refresh only a matching live instance/revision. A false response needs
    // full registration; it never revives an expired or replaced process.
    bool heartbeat(std::string const& instance, uint64_t capacity_revision);
    void worker_admission(std::string const& instance, WorkerAdmission);
    BrokerStatusPage status(StatusRequest const& = {});
    // Worker calls begin before admitting work and compares the direct TLS
    // owner's certificate identity. Only that worker can finish running work.
    bool begin(Lease const&, std::string const& direct_owner_identity);
    void finish(Lease const&, std::optional<CapacityUpdate> const& update = {});
    // Owner reports successful receipt and validation, independently of local
    // durable plot commit. Failure to report must not discard accepted data.
    void delivered(Lease const&, DeliveryObservation const&);
    // Worker rejects unsupported admission before starting compute or output.
    // A retry of this logical task skips that concrete worker/backend pair.
    void reject(Lease const&, FailureCode, std::optional<CapacityUpdate> const& update = {});
    // Owner abandons a failed direct connection. Running reservations remain
    // held until worker finish/expiry; retry skips the failed worker instance.
    void abandon(Lease const&);
    ControlMetrics metrics() const;
    static uint64_t memory_bytes(unsigned channels = control_channel_limit);
private:
    struct Connections;
    std::vector<uint8_t> call(Writer const&, std::string* peer_identity = nullptr, std::string* peer_node_id = nullptr);
    Endpoint endpoint_;
    TlsOptions tls_;
    std::shared_ptr<Connections> connections_;
};
struct BrokerOptions
{
    Endpoint listen;
    TlsOptions tls;
    unsigned lease_ms = 120000, worker_ms = 30000;
    size_t max_workers = 128, max_owners = 128, max_tasks = 4096;
    unsigned control_threads = 4;
    size_t max_connections = 128;
    unsigned idle_ms = 30000;
    // Relative virtual service: generation members or Pack windows, divided
    // by this stable owner ID's share. This is not a wall-time guarantee.
    std::map<std::string, uint32_t> owner_shares;
};
// In-memory leases are fenced by a fresh restart epoch. Owner plot commits and
// recovery remain local; a lease/transport acknowledgement is never a commit.
// Idle persistent channels occupy a bounded registry, never a control thread.
// A stalled handshake/frame occupies one fixed handler; process() holds no
// network waits. max_connections includes accepted and in-flight channels.
class BrokerServer
{
public:
    explicit BrokerServer(BrokerOptions);
    ~BrokerServer();
    BrokerServer(BrokerServer const&) = delete;
    BrokerServer& operator=(BrokerServer const&) = delete;
    Endpoint endpoint() const;
    // Stop and join the serving thread before destroying this object.
    void serve(std::atomic<bool> const& stop);
    // Useful for embedding in an existing service loop.
    void serve_once();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace group_grow::pool
