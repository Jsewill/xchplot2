#pragma once
#include "host/GroupOptions.hpp"
#include <memory>

namespace group_grow { struct WorkerConfig; class WorkerService; }
namespace pos2gpu
{
// Read-only resolution of saved membership and resource defaults. It neither
// creates a pool nor starts any compute. Called before option validation.
void prepare_group_compute(GroupOptions&);
bool saved_compute_available(std::filesystem::path const& directory = {});
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
