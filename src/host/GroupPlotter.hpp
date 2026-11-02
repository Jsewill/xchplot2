#pragma once

#include "host/BatchPlotter.hpp"
#include "host/GroupOptions.hpp"
#include <string_view>

namespace pos2gpu {

// Shared storage and RAM admission for fixed manifests, growth and filling drives.
std::array<uint8_t, 32> compute_owner_seed(std::array<uint8_t, 32> const& seed,
                                           std::string_view owner);
int run_group(GroupOptions const& options);
BatchResult run_group_batch(std::vector<BatchEntry> const& entries,
                            BatchOptions const& options, GroupOptions group_options);

} // namespace pos2gpu
