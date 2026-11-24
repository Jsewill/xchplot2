// Persistent compute execution. Plot identities and durable files stay with the owner.
#pragma once
#include "host/GroupPacked.hpp"
#include "host/ComputePool.hpp"

#include <atomic>
#include <array>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>

namespace group_grow
{
class CodecScheduler;
struct WorkerStatus;
struct WorkerTaskAdmission;
namespace worker_detail { struct RetryHandle; class TaskRegistry; }

class WorkerTaskTimeout : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

struct WorkerCancellation
{
    std::atomic<bool> const* stop = nullptr;
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
    std::atomic<bool> const* service_stop = nullptr;
    std::atomic<bool> const* upstream_stop = nullptr;
    void check() const;
};

struct WorkerContribution
{
    std::span<uint8_t const> bytes;
    uint16_t members = 0;
};
struct WorkerChunkInput
{
    uint64_t ordinal = 0;
    std::span<uint8_t const> old_chunk;
    uint16_t old_members = 0;
    std::span<WorkerContribution const> contributions;
    std::span<std::span<uint64_t const> const> new_members;
};
using WorkerPackedSink = std::function<void(PackedFragmentTile const&, uint64_t total_count)>;
using WorkerFreshManifest = std::function<void(uint64_t payload_bound, uint64_t largest_chunk,
    std::array<uint8_t, 32> const& members_digest)>;
using WorkerContributionSink = std::function<void(uint64_t ordinal,
    std::span<uint8_t const> encoded, uint16_t members)>;
// Borrowed spans remain valid only during the callback. A transport frame is
// delivered in bounded windows without extending its lifetime or copying data.
inline constexpr size_t contribution_window_chunks = 1024;
using WorkerContributionWindowSink = std::function<void(uint64_t ordinal,
    std::span<std::span<uint8_t const> const> encoded, uint16_t members)>;
struct WorkerTaskIdentity
{
    std::string task_id;
    uint64_t attempt = 1;
    // Retain one identity across retries of a logical operation. Copies pin
    // its sequence even while the client's bounded completed-task cache turns
    // over. A new owner session fences the old sequence automatically.
    WorkerTaskIdentity(std::string task = {}, uint64_t version = 1);
private:
    std::shared_ptr<worker_detail::RetryHandle> retry_;
    friend class worker_detail::TaskRegistry;
};
using WorkerEncodedValidator = std::function<void(std::span<std::vector<uint8_t> const>)>;
struct WorkerGenerationTiming
{
    uint64_t worker_ns = 0, send_ns = 0, frame_wait_ns = 0, receiver_ns = 0;
    // Time reading frames after first output, including reads overlapping the sink.
    uint64_t frame_read_ns = 0;
};
struct WorkerGenerationCapacity
{
    uint32_t batch = 0, slots = 0;
    pool::BrokerClient::GenerationAvailability availability;
    uint64_t control_ns = 0;
    std::shared_ptr<pool::GenerationRoster const> roster;
};
struct WorkerPackedRoute
{
    pool::GenerationRoute route;
    pool::Kind kind = pool::Kind::GenerateAny;
    std::shared_ptr<void> reserved_singleton;
    struct Placement;
    std::shared_ptr<Placement> placement;
    unsigned paid_width = 0;
    bool matches(pool::Lease const& lease) const
    {
        return kind == lease.task.kind && route.identity == lease.worker_identity &&
            route.instance == lease.worker_instance && route.endpoint == lease.endpoint;
    }
};
struct WorkerGenerationEstimate
{
    long double member_ns = 0;
    // The selected receiver count and all slots represented by the measurements.
    unsigned receivers = 0, measured_slots = 0;
    bool dated_prior = false;
    unsigned prior_min_samples = 0;
    uint64_t prior_oldest_utc = 0;
    bool measured_packed_batch = false;
};
struct WorkerStats
{
    uint64_t input_bytes = 0, output_bytes = 0;
    uint64_t completed_generation = 0, completed_contributions = 0, completed_codec = 0, failed_tasks = 0;
    uint64_t local_tasks = 0, task_elapsed_ns = 0;
    uint64_t peak_generation_members = 0, peak_active_generation = 0;
    uint64_t timed_generation_tasks = 0;
    WorkerGenerationTiming generation;
    uint64_t independent_tasks = 0, independent_members = 0;
};
struct WorkerTaskObservation
{
    std::string worker_identity, worker_instance;
    pool::Kind kind = pool::Kind::GenerateCpu;
    uint8_t k = 0, strength = 0;
    // Owner wall phases: selection to first output, then result consumption.
    // Include admission, transport, sinks and acknowledgement, not kernel or
    // pure network timing. Local generation uses the same output boundary.
    uint64_t work_units = 0, elapsed_ns = 0, execution_ns = 0, transfer_ns = 0;
    uint64_t input_bytes = 0, output_bytes = 0;
    bool local_direct = false;
    pool::GenerationOutput generation_output = pool::GenerationOutput::Packed;
    // Singleton slots for this workload in the worker's advertised capacity.
    unsigned worker_slots = 0;
    WorkerGenerationTiming generation;
    bool independent_members = false;
    uint32_t ingress_interface = 0; // Owner-local physical adapter, zero if unresolved.
    uint64_t connect_ns = 0;
    // Successful generation receipts identify the exact public plot IDs,
    // including singleton requests coalesced across different destinations.
    std::array<std::array<uint8_t, 32>, pool::max_generation_batch_members> member_ids{};
    unsigned member_id_count = 0;
    pool::Resources generation_resources{};
    std::optional<int> local_device; // Display selector for a private in-process executor.
};
// Exact validated result for a proposed single-lane whole-group publication.
struct WorkerPublication
{
    pool::PublicationRoute route;
    std::string path, host, gpu;
    bool loopback = false;
    std::vector<std::array<uint8_t, 32>> ids;
    std::atomic<unsigned> deliveries{0};
    uint8_t k = 0, strength = 0;
    std::atomic<bool> encoded{false}, promoted{false};
    bool selected_fresh = false;
    std::function<bool()> pressure_unchanged;
    bool matches(pool::Lease const& lease) const
    {
        return lease.worker_identity == route.identity && lease.worker_instance == route.instance && lease.endpoint == route.endpoint &&
            lease.task.kind == route.kind && lease.task.resources == route.resources && lease.task.work_units == ids.size() &&
            lease.task.k == k && lease.task.strength == strength;
    }
    void delivered(WorkerTaskObservation const& value, pool::Endpoint const& endpoint, std::string const& actual_path)
    {
        if (value.kind == pool::Kind::Pack) return;
        bool const matched = !value.local_direct && !value.independent_members && value.generation.worker_ns &&
            value.worker_identity == route.identity && value.worker_instance == route.instance && endpoint == route.endpoint &&
            value.kind == route.kind && value.generation_resources == route.resources && actual_path == path &&
            value.k == k && value.strength == strength && value.work_units == ids.size() && value.member_id_count == ids.size() && std::equal(ids.begin(), ids.end(), value.member_ids.begin()) &&
            (value.generation_output == pool::GenerationOutput::Packed || value.generation_output == pool::GenerationOutput::Fresh);
        if (matched) encoded.store(value.generation_output == pool::GenerationOutput::Fresh);
        deliveries.fetch_add(matched ? 1 : 2);
    }
};
struct WorkerGenerationPrior
{
    uint64_t compute_low = 0, compute_high = 0, wire_low = 0, wire_high = 0;
    std::chrono::seconds remaining{};
    unsigned samples = 0;
    uint64_t oldest_utc = 0;
};
using WorkerGenerationCalibration = std::function<WorkerGenerationPrior(WorkerTaskObservation const&,
    pool::GenerationRoute const&, uint64_t compute_ns_per_member, uint64_t wire_ns_per_member)>;
class WorkerTaskDeclined : public std::runtime_error
{
public:
    WorkerTaskDeclined() : std::runtime_error("optional compute task declined for selected worker") {}
};
class WorkerPublicationChanged : public WorkerTaskDeclined {};
class WorkerGenerationFailed : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
struct WorkerTaskOptions
{
    std::function<bool(WorkerTaskObservation const&)> selected;
    std::function<void(WorkerTaskObservation const&)> observe;
    // Before a generation lease starts, the broker can report a supported
    // independent width. False withdraws the owner's queued request.
    std::function<bool(unsigned)> queued_generation;
    // Expected completion of other known work on independent destinations.
    std::chrono::steady_clock::time_point other_work_until{};
    WorkerPackedRoute const* packed_route = nullptr;
    bool fresh_read_ahead = true;
    bool immediate_fresh = false;
    std::shared_ptr<WorkerPublication> publication;
};
class WorkerAlreadyRunning : public std::runtime_error
{
public:
    WorkerAlreadyRunning() : std::runtime_error("another compute service already offers this machine's identity") {}
};
struct WorkerIdentityReservation;

// Compatibility covers canonical generation and encoding, independently of
// machine/backend selection and the owner's persisted construction identity.
std::string worker_compatibility(pool::Kind = pool::Kind::GenerateAny);
std::string worker_producer_build();
uint64_t worker_cpu_generation_bytes(uint8_t k, uint8_t strength, unsigned threads = 0);

struct WorkerConfig
{
    pool::Endpoint broker, listen, advertise;
    // All brokers share this executor's one CPU/RAM/GPU/slot admission gate.
    // Discovery results are untrusted routes until mutual TLS registration.
    std::vector<pool::Endpoint> brokers;
    std::function<std::vector<pool::Endpoint>()> discover_brokers;
    // Persistent control-channel storage is reserved against host capacity.
    // Each refresh visits a rotating bounded batch concurrently, so failed
    // brokers cannot serialize delays for the rest of the discovered pool.
    size_t max_brokers = 128;
    unsigned broker_refresh_parallelism = 8, broker_refresh_ms = 500;
    unsigned broker_rotation_ms = 30000;
    std::string server_name;
    std::shared_ptr<WorkerIdentityReservation> identity_reservation;
    pool::TlsOptions tls;
    pool::Resources capacity;
    std::string backend = "cpu", tier = "auto";
    int device = 0;
    bool gpu_capacity_measured = false;
    // Optional CPU/NUMA selection. CPU generation restores the caller's mask
    // after scoped pinning; GPU runtime/codec threads keep their own policy.
    std::vector<int> cpu_affinity;
    bool require_cpu_affinity = false;
    unsigned generation_threads = 0, max_batch_members = 4, max_window_chunks = pool::default_max_codec_chunks;
    unsigned task_timeout_ms = 3600000;
    // Codec input/output windows and count-derived initial-contribution scratch.
    uint64_t max_task_bytes = pool::default_codec_task_bytes;
    // Embedded construction and compute share this weighted CPU gate. Host
    // capacity is a separate explicit partition, including idle transport and
    // retained GPU allocations. An optional embedding reservation owns it.
    std::shared_ptr<CodecScheduler> cpu_scheduler;
    std::shared_ptr<void> host_reservation;
    // Private device lanes share their owner's RAM partition. A recovery
    // probe must acquire a real reservation alongside already granted jobs.
    std::function<std::shared_ptr<void>(uint64_t)> reserve_gpu_probe;
    // Optional embedding policy, called with finalized configuration before
    // service/GPU executor construction. Throws before unreviewed work starts.
    std::function<void(WorkerConfig const&)> admit_start;
    // Optional embedding observation, bounded again by capacity. The CPU
    // argument is this executor's currently held lanes; memory is fresh free
    // headroom before its outstanding modeled allocations are charged.
    std::function<pool::Resources(uint32_t)> available_resources;
    std::optional<std::filesystem::path> local_harvester;
    unsigned farming_deadline_ms = 0;
};

enum class WorkerControl : uint8_t { Pause = 1, Drain, Resume };
struct WorkerStatus
{
    std::string identity, node_id, instance;
    std::string compatibility, codec_compatibility, producer_build;
    // Saved enrollment owns admission intent across standalone and automatic
    // executors. Direct unmanaged TLS controls last for this process only.
    bool durable_control = false;
    pool::WorkerAdmission admission = pool::WorkerAdmission::Accepting;
    pool::Resources capacity, active{0, 0, 0, 0};
    uint64_t accepted_tasks = 0, waiting_cpu = 0, baseline_host_bytes = 0;
    uint64_t broker_limit = 0, known_brokers = 0, healthy_brokers = 0, deferred_brokers = 0;
    uint64_t broker_transport_bytes = 0;
    bool broker_saturated = false;
    std::string gpu_recovery;
    pool::WorkerHardware hardware;
    std::optional<bool> farming_pressure;
};
struct WorkerTaskAdmission
{
    WorkerStatus worker;
    pool::Task task;
    std::string owner_identity, owner_id, broker_identity;
    bool local_fallback = false;
};
struct WorkerLocalOffer
{
    pool::Resources capacity, demand{0, 0, 0, 0};
    uint64_t standing_host_bytes = 0, member_host_bytes = 0;
    unsigned max_batch_members = 0;
    std::string gpu_recovery;
    bool gpu_headroom_limited = false;
};
class WorkerAdmissionRejected : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
class WorkerResourcesUnavailable : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class WorkerService
{
public:
    explicit WorkerService(WorkerConfig);
    ~WorkerService();
    WorkerService(WorkerService const&) = delete;
    WorkerService& operator=(WorkerService const&) = delete;
    void start();
    void stop();
    // Reserve before constructing a GPU executor, so concurrent plotters can
    // reuse an existing service without allocating a second local producer.
    static std::shared_ptr<WorkerIdentityReservation> reserve_identity(std::string const& certificate);
    pool::Endpoint endpoint() const;
    bool owns(pool::Lease const&) const;
    WorkerStatus status() const;
    WorkerLocalOffer local_generation_offer(uint8_t k, uint8_t strength, unsigned members) const;
    WorkerStatus control(WorkerControl);
    static WorkerStatus remote_status(pool::Endpoint const&, pool::TlsOptions const&);
    // Mutations require the running worker's authenticated machine identity.
    static WorkerStatus remote_control(pool::Endpoint const&, pool::TlsOptions const&, WorkerControl);
    // Same executor and admission as remote calls, including during broker
    // outages. No independent fallback producer may bypass these resources.
    void generate_local(uint8_t k, uint8_t strength,
        std::array<uint8_t, 32> const&, WorkerPackedSink const&, WorkerCancellation = {},
        pool::Kind = pool::Kind::GenerateAny, std::function<void(WorkerTaskAdmission const&)> const& = {});
    void generate_packed_local(uint8_t, uint8_t, std::span<std::array<uint8_t, 32> const>,
        WorkerPackedSink const&, WorkerCancellation, pool::Kind,
        std::function<void(WorkerTaskAdmission const&)> const&, pool::Resources const* = nullptr);
    void contributions_local(uint8_t k, uint8_t strength,
        std::span<std::array<uint8_t, 32> const>, WorkerContributionSink const&,
        WorkerCancellation = {}, pool::Kind = pool::Kind::GenerateAny,
        std::function<void(WorkerTaskAdmission const&)> const& = {}, pool::Resources const* = nullptr);
private:
    friend class WorkerClient;
    unsigned generate_assigned(pool::Lease const&, std::span<std::array<uint8_t, 32> const>,
        WorkerPackedSink const&, WorkerContributionSink const&, WorkerContributionWindowSink const&,
        WorkerFreshManifest const&, WorkerCancellation);
    void encode_assigned(pool::Lease const&, std::span<uint8_t const>, uint64_t,
        std::function<void(size_t, std::span<uint8_t const>)> const&, WorkerCancellation);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
int run_worker_gpu_child(char const* handle, char const* bytes);
void default_worker_resources(WorkerConfig&);
// Complete startup service/transport and selected GPU runtime host allowance,
// inside capacity.host_bytes rather than additional to that total partition.
// An explicitly shared scheduler must also cover the complete offered CPUs.
uint64_t worker_service_baseline_bytes(WorkerConfig const&);
uint64_t worker_cpu_task_bytes(uint8_t k, uint8_t strength, uint64_t members, uint64_t transport_bytes, unsigned threads = 0);
// Automatic services may reduce their discovery ceiling to retain room for a
// supported CPU generation task. Explicit limits remain constructor errors.
void fit_worker_broker_budget(WorkerConfig&);

// A private owner keeps its shared host/CPU grant until the last local sink
// returns, including when admission, generation or a consumer throws.
struct WorkerLocalGrant
{
    WorkerService* service = nullptr;
    pool::Resources resources;
    bool delivered = false;
    std::optional<int> local_device;
    virtual ~WorkerLocalGrant() = default;
};
struct WorkerClientConfig
{
    pool::Endpoint broker;
    pool::TlsOptions tls;
    std::string owner_id, compatibility, codec_compatibility;
    bool ephemeral_owner = false;
    // Private local executors use the same admission and cancellation gates
    // without a broker, TLS identity, listener or enrollment. Encoding remains
    // an owner codec responsibility in this mode.
    bool local_only = false;
    pool::Kind generation_kind = pool::Kind::GenerateAny;
    unsigned generation_threads = 0, max_window_chunks = pool::default_max_codec_chunks;
    uint64_t max_task_bytes = pool::default_codec_task_bytes;
    unsigned task_timeout_ms = 3600000;
    std::shared_ptr<CodecScheduler> cpu_scheduler;
    bool packed_receive_overlap = true;
    // The embedding owner keeps the service alive through every client call.
    WorkerService* local_service = nullptr;
    std::function<std::shared_ptr<WorkerService>()> local_service_provider;
    // A queued broker request may take an immediately available private grant.
    // Local-only work and explicit outage fallback can wait for a grant.
    std::function<std::unique_ptr<WorkerLocalGrant>(uint8_t, uint8_t, unsigned, pool::GenerationOutput,
        WorkerCancellation const&, bool wait)> acquire_private_generation;
    std::function<WorkerGenerationCapacity(uint8_t, uint8_t)> private_generation_capacity;
    std::function<void(WorkerTaskObservation const&)> observe;
    std::function<void(WorkerTaskObservation const&, std::string_view)> generation_failed;
    // The plotting client's normal cached broker query publishes changed
    // eligible routes. No network request is needed by a display observer.
    std::function<void(uint8_t, uint8_t, WorkerGenerationCapacity const&)> generation_capacity_changed;
    WorkerGenerationCalibration generation_calibration;
    // The plotting host and its receive policy bind dated observations across
    // logical jobs. The source address and adapter come from the actual data route.
    std::function<std::string(pool::GenerationRoute const&, std::string const&, uint32_t)> generation_path;
    // Called after selection, before worker contact. A caller may reject an
    // optional task by throwing; the unbegun grant is then cancelled.
    std::function<void(WorkerTaskObservation const&)> selected;
    // Whole-service status is authenticated separately before contacting the
    // selected executor with Work. Local outage fallback uses the same hook
    // after resolving its concrete resource grant, before allocation/compute.
    std::function<void(WorkerTaskAdmission const&)> admit_task;
    // Shared ceiling for queued/acquiring and executing generation requests.
    unsigned max_active_generation = 64;
    // The owner reserves one extra receive buffer for each concurrent task.
    bool contribution_read_ahead = false;
};

// One client is safe for concurrent requests. Each admitted task keeps a
// direct worker TLS session for all its bounded input/output windows.
class WorkerClient
{
public:
    explicit WorkerClient(WorkerClientConfig);
    ~WorkerClient();
    WorkerClient(WorkerClient const&) = delete;
    WorkerClient& operator=(WorkerClient const&) = delete;
    // True only if this client checked all packed fields and their narrow deltas on receipt.
    bool generate(uint8_t k, uint8_t strength, std::array<uint8_t, 32> const& plot_id,
                  WorkerPackedSink const&, WorkerCancellation = {}, WorkerTaskIdentity = {},
                  std::chrono::steady_clock::time_point other_work_until = {});
    // A funded multi-receiver owner can reclaim the untouched range when a
    // paid route changes before Work. Otherwise this sink stays ordered.
    bool generate_packed(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> plot_ids,
        WorkerPackedSink const&, WorkerCancellation = {}, WorkerTaskIdentity = {},
        std::chrono::steady_clock::time_point other_work_until = {}, WorkerPackedRoute const* batch_route = nullptr,
        bool* reprice_unstarted = nullptr);
    void contributions(uint8_t k, uint8_t strength,
        std::span<std::array<uint8_t, 32> const> plot_ids,
        WorkerContributionSink const&, WorkerCancellation = {}, WorkerTaskIdentity = {},
        WorkerContributionWindowSink const& = {}, WorkerFreshManifest const& = {}, bool independent = false,
        bool coalesce_singletons = true, std::chrono::steady_clock::time_point other_work_until = {},
        bool fresh_read_ahead = true, bool immediate_fresh = false);
    // Zero means all members were delivered. A positive width releases the
    // unstarted lease for independent delivery of that prefix and its remainder.
    unsigned try_contributions(uint8_t k, uint8_t strength,
        std::span<std::array<uint8_t, 32> const> plot_ids, WorkerContributionSink const&,
        WorkerCancellation, WorkerTaskIdentity, WorkerContributionWindowSink const&, bool independent,
        std::chrono::steady_clock::time_point other_work_until = {});
    // Outage fallback shares the same admission ceiling as broker tasks. The
    // embedding owner records fallback completions separately from pool stats.
    void generate_local(uint8_t k, uint8_t strength, std::array<uint8_t, 32> const&,
        WorkerPackedSink const&, WorkerCancellation = {});
    void contributions_local(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const>,
        WorkerContributionSink const&, WorkerCancellation = {});
    std::vector<std::vector<uint8_t>> encode(uint8_t k,
        std::span<WorkerChunkInput const> chunks, uint64_t max_output_bytes,
        WorkerCancellation = {}, WorkerTaskIdentity = {}, WorkerEncodedValidator const& = {},
        WorkerTaskOptions const& = {});
    WorkerStats stats() const;
    bool local_available() const;
    unsigned max_window_chunks() const;
    uint64_t max_task_bytes() const;
    std::shared_ptr<WorkerPublication> publication_route(uint8_t k, uint8_t strength,
        std::span<std::array<uint8_t, 32> const> ids) const;
    // Remote-only formats query broker capacity without private generation slots.
    WorkerGenerationCapacity generation_capacity(uint8_t k, uint8_t strength, bool include_private = true) const;
    // Recheck a previously withdrawn batch against a complete signed price at its exact width.
    std::optional<WorkerPackedRoute> priced_packed_route(uint8_t k, uint8_t strength, unsigned width,
        WorkerCancellation cancel) const;
    unsigned packed_generation_width(uint8_t k, uint8_t strength, unsigned ready, unsigned destinations,
                                     WorkerPackedRoute* batch_route = nullptr, unsigned* singleton_limit = nullptr,
                                     std::chrono::steady_clock::time_point other_work_until = {}) const;
    // Measured whole elapsed time of the next eligible batch or singleton wave.
    // An incomplete live route or width has no price.
    std::optional<long double> publication_trial_ns(uint8_t k, uint8_t strength, pool::GenerationOutput output, unsigned members,
        bool joint_packed, WorkerGenerationCapacity const& capacity) const;
    unsigned independent_contribution_width(uint8_t k, uint8_t strength, unsigned ready_members = 1,
        unsigned destinations = 1, bool dispatch = false) const;
    // A zero joint width selects the independent pipeline for this ready work.
    unsigned joint_contribution_members(uint8_t k, uint8_t strength, unsigned ready_members, unsigned destination_batch = 0) const;
    bool probe_contribution_route(uint8_t k, uint8_t strength) const;
    bool codec_probe_available(uint8_t k, uint64_t timeout_ns) const;
    unsigned suggested_batch_members(uint8_t k, uint8_t strength, unsigned maximum,
        pool::GenerationOutput output = pool::GenerationOutput::Packed) const;
    unsigned suggested_generation_concurrency(uint8_t k, uint8_t strength, unsigned maximum,
        pool::GenerationOutput output = pool::GenerationOutput::Packed) const;
    // A positive ready-work horizon uses first deliveries to size receivers.
    WorkerGenerationEstimate generation_estimate(uint8_t k, uint8_t strength, pool::GenerationOutput output,
        unsigned active, unsigned batch, uint64_t horizon_members = 0) const;
    static uint64_t memory_bytes();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace group_grow
