#pragma once
#include "host/GroupOptions.hpp"
#include <chrono>
#include <memory>

namespace group_grow { struct WorkerConfig; class WorkerService; class CodecScheduler; }
namespace pos2gpu
{
// Read-only resolution of saved membership and resource defaults. It neither
// creates a pool nor starts any compute. Called before option validation.
void prepare_group_compute(GroupOptions&);
bool saved_compute_available(std::filesystem::path const& directory = {});
// Read-only construction of the same finalized offer used by the backend.
// Saved membership, broker transport limits and default resources are resolved;
// no listener, worker, discovery service or GPU executor is started.
group_grow::WorkerConfig make_group_compute_offer(GroupOptions const&,
    std::shared_ptr<group_grow::CodecScheduler> = {});
struct ComputeInspection
{
    struct Issue { std::string endpoint, reason; };
    std::string status = "not-queried", reason, next_action, unknowns, discovery = "not-queried";
    std::string certificate_expires, authority_expires, service_lifecycle, local_worker_intent = "not-selected", local_worker_intent_action;
    std::vector<Issue> issues;
    uint64_t observed_at_unix_ms = 0, brokers = 0, workers = 0, oldest_worker_registration_age_ms = 0;
    uint64_t accepting_generation = 0, accepting_codec = 0, paused = 0, draining = 0, drained = 0, incompatible = 0;
    bool configured = false, checked = false, complete = true, blocking = false;
    bool certificate_expired = false, authority_expired = false;
};
// Bounded advisory inspection only: saved membership, numeric broker routes,
// authenticated worker registrations and admission intent. No owner session,
// lease, service, saved-policy import or compute executor is created. The
// caller shares one deadline with any other live diagnostics.
ComputeInspection inspect_group_compute(GroupOptions const&, std::chrono::steady_clock::time_point deadline);
class ComputeSession
{
public:
    // Assigns the active broker and execution session to these options.
    explicit ComputeSession(GroupOptions&);
    ~ComputeSession();
    ComputeSession(ComputeSession const&) = delete;
    ComputeSession& operator=(ComputeSession const&) = delete;
    void offer_local(group_grow::WorkerConfig);
    std::shared_ptr<group_grow::WorkerService> local_worker() const;
    // Total partition, including service baseline, only while this session
    // owns the worker identity and its startup/live allocations.
    uint64_t owned_worker_bytes() const;
    static constexpr uint64_t memory_bytes = 128ULL << 20;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace pos2gpu
