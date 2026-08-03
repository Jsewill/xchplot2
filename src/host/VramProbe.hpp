// VramProbe.hpp — validation for a Level Zero Sysman memory reading.
//
// Header-only and free of SYCL/CUDA/Level Zero so vram_probe_test can exercise
// it on any machine. That is the entire point: the code it guards runs only on
// Intel hardware, the ABI it consumes is hand-declared and cannot be
// compile-checked, and a wrong answer here steers a tier past what the card
// physically has. Every branch below is unreachable on a CUDA dev box.
//
// Sysman measures against PHYSICAL memory. global_mem_size is what the runtime
// will actually hand out. They are NOT the same number, and free can
// legitimately exceed the smaller one. Measured on an Arc B580:
//
//   sysman size       12216 MiB   (physical)
//   sysman free       11688 MiB   (idle; 11610 a moment later, tracking)
//   global_mem_size   11605 MiB   (allocatable — 611 MiB less than physical)
//
// The first version of this bounded free by global_mem_size and rejected every
// reading on that card. The ABI was right; the bound was wrong. Hence the test.

#pragma once

#include <algorithm>
#include <cstdint>

namespace pos2gpu {

struct VramReading {
    std::uint64_t free  = 0;
    std::uint64_t total = 0;
};

// Validate a sysman (free, total) pair against the device's own global memory
// size, and normalise it into the allocatable space the tier picker works in.
//
// `global_mem` may be 0 when the caller could not determine it, in which case
// sysman's own figures are taken as-is.
//
// Returns false when the reading cannot be explained — that must degrade to
// "no probe" (the caller falls back to the device total), never to a
// plausible-looking number.
inline bool validate_sysman_reading(std::uint64_t sysman_free,
                                    std::uint64_t sysman_total,
                                    std::uint64_t global_mem,
                                    VramReading&  out)
{
    // Internal consistency first: a misparsed struct gives garbage, and
    // garbage almost never satisfies free <= total with a non-zero total.
    if (sysman_total == 0 || sysman_free > sysman_total) return false;

    // Then loose agreement with the device's own figure. Physical and
    // allocatable differ by a few percent in practice (B580: 1.05x), so 2x
    // either way is generous while still catching a field read from the wrong
    // offset, which lands orders of magnitude out.
    if (global_mem != 0
        && (sysman_total < global_mem / 2 || sysman_total > global_mem * 2)) {
        return false;
    }

    // CLAMP to the allocatable size rather than reporting sysman's figure.
    // Free is relative to physical, so an idle card reports more free than can
    // actually be allocated; handing the picker those extra 611 MiB would be
    // precisely the over-commit this probe exists to prevent. Clamped, the
    // number still tracks real usage downward, which is what was missing.
    out.total = (global_mem != 0) ? global_mem : sysman_total;
    out.free  = std::min(sysman_free, out.total);
    return true;
}

} // namespace pos2gpu
