#pragma once
#include "host/GroupOptions.hpp"
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
    static constexpr uint64_t memory_bytes = 128ULL << 20;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace pos2gpu
