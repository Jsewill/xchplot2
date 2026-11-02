// Persistent compute execution. Plot identities and durable files stay with the owner.
#pragma once
#include "host/GroupPacked.hpp"
#include "host/ComputePool.hpp"

#include <atomic>
#include <array>
#include <chrono>
#include <functional>
#include <memory>

namespace group_grow
{
class CodecScheduler;
namespace worker_detail { struct RetryHandle; class TaskRegistry; }

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
using WorkerContributionSink = std::function<void(uint64_t ordinal,
    std::span<uint8_t const> encoded, uint16_t members)>;
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
struct WorkerStats
{
    uint64_t input_bytes = 0, output_bytes = 0;
    uint64_t completed_generation = 0, completed_contributions = 0, completed_codec = 0, failed_tasks = 0;
    uint64_t local_tasks = 0, task_elapsed_ns = 0;
    uint64_t peak_generation_members = 0, peak_active_generation = 0;
};
struct WorkerTaskObservation
{
    std::string worker_identity, worker_instance;
    pool::Kind kind = pool::Kind::GenerateCpu;
    uint8_t k = 0, strength = 0;
    // Owner-observed phases: selection to first output, then result receipt.
    // These include scheduling/transport latency, rather than kernel timing.
    uint64_t work_units = 0, elapsed_ns = 0, execution_ns = 0, transfer_ns = 0;
    uint64_t input_bytes = 0, output_bytes = 0;
    bool local_direct = false;
};
class WorkerTaskDeclined : public std::runtime_error
{
public:
    WorkerTaskDeclined() : std::runtime_error("optional compute task declined for selected worker") {}
};
struct WorkerTaskOptions
{
    std::function<bool(WorkerTaskObservation const&)> selected;
    std::function<void(WorkerTaskObservation const&)> observe;
};
class WorkerAlreadyRunning : public std::runtime_error
{
public:
    WorkerAlreadyRunning() : std::runtime_error("another compute service already offers this machine's identity") {}
};
struct WorkerIdentityReservation;

// Compatibility covers canonical generation and encoding, independently of
// machine/backend selection and the owner's persisted construction identity.
std::string worker_compatibility();
uint64_t worker_cpu_generation_bytes(uint8_t k, uint8_t strength);

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
    unsigned generation_threads = 0, max_batch_members = 4, max_window_chunks = pool::default_max_codec_chunks;
    unsigned task_timeout_ms = 3600000;
    // Codec input/output windows and count-derived initial-contribution scratch.
    uint64_t max_task_bytes = pool::default_codec_task_bytes;
    // Embedded construction and compute share this weighted CPU gate. Host
    // capacity is a separate explicit partition, including idle transport and
    // retained GPU allocations. An optional embedding reservation owns it.
    std::shared_ptr<CodecScheduler> cpu_scheduler;
    std::shared_ptr<void> host_reservation;
};

enum class WorkerControl : uint8_t { Pause = 1, Drain, Resume };
struct WorkerStatus
{
    std::string identity, node_id, instance;
    pool::WorkerAdmission admission = pool::WorkerAdmission::Accepting;
    pool::Resources capacity, active{0, 0, 0, 0};
    uint64_t accepted_tasks = 0, waiting_cpu = 0, baseline_host_bytes = 0;
    uint64_t broker_limit = 0, known_brokers = 0, healthy_brokers = 0, deferred_brokers = 0;
    uint64_t broker_transport_bytes = 0;
    bool broker_saturated = false;
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
    WorkerStatus control(WorkerControl);
    static WorkerStatus remote_status(pool::Endpoint const&, pool::TlsOptions const&);
    // Mutations require the running worker's authenticated machine identity.
    static WorkerStatus remote_control(pool::Endpoint const&, pool::TlsOptions const&, WorkerControl);
    // Same executor and admission as remote calls, including during broker
    // outages. No independent fallback producer may bypass these resources.
    void generate_local(uint8_t k, uint8_t strength,
        std::array<uint8_t, 32> const&, WorkerPackedSink const&, WorkerCancellation = {},
        pool::Kind = pool::Kind::GenerateAny);
    void contributions_local(uint8_t k, uint8_t strength,
        std::span<std::array<uint8_t, 32> const>, WorkerContributionSink const&,
        WorkerCancellation = {}, pool::Kind = pool::Kind::GenerateAny);
private:
    friend class WorkerClient;
    void generate_assigned(pool::Lease const&, std::span<std::array<uint8_t, 32> const>,
        WorkerPackedSink const&, WorkerContributionSink const&, WorkerCancellation);
    void encode_assigned(pool::Lease const&, std::span<uint8_t const>, uint64_t,
        std::function<void(size_t, std::span<uint8_t const>)> const&, WorkerCancellation);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
void default_worker_resources(WorkerConfig&);
// Automatic services may reduce their discovery ceiling to retain room for a
// supported CPU generation task. Explicit limits remain constructor errors.
void fit_worker_broker_budget(WorkerConfig&);

struct WorkerClientConfig
{
    pool::Endpoint broker;
    pool::TlsOptions tls;
    std::string owner_id, compatibility;
    bool ephemeral_owner = false;
    pool::Kind generation_kind = pool::Kind::GenerateAny;
    unsigned generation_threads = 0, max_window_chunks = pool::default_max_codec_chunks;
    uint64_t max_task_bytes = pool::default_codec_task_bytes;
    unsigned task_timeout_ms = 3600000;
    std::shared_ptr<CodecScheduler> cpu_scheduler;
    // The embedding owner keeps the service alive through every client call.
    WorkerService* local_service = nullptr;
    std::function<std::shared_ptr<WorkerService>()> local_service_provider;
    std::function<void(WorkerTaskObservation const&)> observe;
    // Called after selection, before worker contact. A caller may reject an
    // optional task by throwing; the unbegun grant is then cancelled.
    std::function<void(WorkerTaskObservation const&)> selected;
    // Shared ceiling for queued/acquiring and executing generation requests.
    unsigned max_active_generation = 64;
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
    void generate(uint8_t k, uint8_t strength, std::array<uint8_t, 32> const& plot_id,
                  WorkerPackedSink const&, WorkerCancellation = {}, WorkerTaskIdentity = {});
    void contributions(uint8_t k, uint8_t strength,
        std::span<std::array<uint8_t, 32> const> plot_ids,
        WorkerContributionSink const&, WorkerCancellation = {}, WorkerTaskIdentity = {});
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
    unsigned suggested_batch_members(uint8_t k, uint8_t strength, unsigned maximum) const;
    unsigned suggested_generation_concurrency(uint8_t k, uint8_t strength, unsigned maximum) const;
    static uint64_t memory_bytes();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace group_grow
