// Generate a single CPU member and atomically publish it through the shared writer.
#pragma once

#include <cstdint>

namespace pos2gpu {

struct BatchEntry;
struct BatchOptions;

// Returns the on-disk plot size in bytes (via file_size post-write).
std::uint64_t run_one_plot_cpu(BatchEntry const& entry, BatchOptions const& opts);

} // namespace pos2gpu
