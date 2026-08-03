// vram_probe_test — validation of a Level Zero Sysman memory reading.
//
// The guarded code runs only on Intel hardware and parses a hand-declared ABI,
// so on a CUDA dev box every branch is unreachable and "it compiles" is the
// only feedback available. That is exactly how the first version shipped a
// bound that rejected every reading on the target card.
//
// The B580 case below is the real measurement from that host, so the specific
// mistake that cost a round trip cannot come back.
//
// Pure integer arithmetic — no SYCL, CUDA, Level Zero, or device.

#include "host/VramProbe.hpp"

#include <cstdint>
#include <cstdio>

namespace {

int failures = 0;

constexpr std::uint64_t MiB = 1024ull * 1024ull;

void check(bool ok, char const* what)
{
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

void check_eq(std::uint64_t got, std::uint64_t want, char const* what)
{
    bool const ok = (got == want);
    std::printf("%s %s (got %llu MiB, want %llu MiB)\n", ok ? "PASS" : "FAIL",
                what, (unsigned long long)(got / MiB),
                (unsigned long long)(want / MiB));
    if (!ok) ++failures;
}

// ---- the measurement this was got wrong on ---------------------------------

void test_arc_b580_idle()
{
    // Intel Arc B580, idle, as reported by POS2GPU_VRAM_PROBE_DEBUG=1:
    //   sysman size 12216 MiB, sysman free 11688 MiB, global_mem_size 11605 MiB
    // Free EXCEEDS global_mem_size because sysman measures physical memory and
    // 611 MiB of the card is not allocatable. Bounding free by global_mem_size
    // rejected this — the ABI was right, the bound was wrong.
    pos2gpu::VramReading r;
    bool const ok = pos2gpu::validate_sysman_reading(
        11688 * MiB, 12216 * MiB, 11605 * MiB, r);
    check(ok, "B580 idle: accepted (free may exceed global_mem_size)");
    check_eq(r.free,  11605 * MiB, "B580 idle: free clamped to allocatable");
    check_eq(r.total, 11605 * MiB, "B580 idle: total is the allocatable figure");

    // A moment later the same card read 11610 MiB free — still above
    // global_mem_size, still legitimate.
    pos2gpu::VramReading r2;
    check(pos2gpu::validate_sysman_reading(11610 * MiB, 12216 * MiB,
                                           11605 * MiB, r2),
          "B580 second sample: accepted");
    check_eq(r2.free, 11605 * MiB, "B580 second sample: clamped");
}

// ---- the case the probe exists for -----------------------------------------

void test_tracks_real_usage()
{
    // Mid-plot, with the pool resident. This is the whole point: the number
    // must move, where global_mem_size alone never does.
    pos2gpu::VramReading r;
    check(pos2gpu::validate_sysman_reading(1200 * MiB, 12216 * MiB,
                                           11605 * MiB, r),
          "busy card: accepted");
    check_eq(r.free, 1200 * MiB, "busy card: free reported as measured");

    // Another tenant holding most of the card.
    pos2gpu::VramReading r2;
    check(pos2gpu::validate_sysman_reading(300 * MiB, 12216 * MiB,
                                           11605 * MiB, r2),
          "contended card: accepted");
    check_eq(r2.free, 300 * MiB, "contended card: no clamp when below cap");
}

// ---- misparse rejection ----------------------------------------------------
// A wrong struct offset is the failure mode with teeth: it yields a
// plausible-looking number that sends a tier past what the card has. These
// must degrade to "no probe", never to a value.

void test_rejects_misparse()
{
    pos2gpu::VramReading r;

    check(!pos2gpu::validate_sysman_reading(0, 0, 11605 * MiB, r),
          "reject: zero total");

    check(!pos2gpu::validate_sysman_reading(20000 * MiB, 12216 * MiB,
                                            11605 * MiB, r),
          "reject: free exceeds sysman's own total");

    // Field read from the wrong offset — lands orders of magnitude out.
    check(!pos2gpu::validate_sysman_reading(1 * MiB, 1ull << 50,
                                            11605 * MiB, r),
          "reject: total nowhere near the device figure (high)");
    check(!pos2gpu::validate_sysman_reading(1 * MiB, 8 * MiB,
                                            11605 * MiB, r),
          "reject: total nowhere near the device figure (low)");

    // A pointer misread as a length is the classic one.
    check(!pos2gpu::validate_sysman_reading(0x7feb87a08000ull, 0x7feb87a08000ull,
                                            11605 * MiB, r),
          "reject: pointer-shaped values");
}

// ---- tolerance boundaries --------------------------------------------------

void test_tolerance_window()
{
    pos2gpu::VramReading r;
    std::uint64_t const cap = 11605 * MiB;

    // Physical exceeding allocatable by 5% is the real-world case and must sit
    // comfortably inside the window, not near its edge.
    check(pos2gpu::validate_sysman_reading(cap, cap + cap / 20, cap, r),
          "tolerance: +5% physical accepted (the B580 case)");

    check(pos2gpu::validate_sysman_reading(cap / 2, cap * 2, cap, r),
          "tolerance: 2x total still accepted");
    check(!pos2gpu::validate_sysman_reading(cap / 2, cap * 2 + MiB, cap, r),
          "tolerance: beyond 2x rejected");
    check(pos2gpu::validate_sysman_reading(cap / 4, cap / 2, cap, r),
          "tolerance: half total still accepted");
    check(!pos2gpu::validate_sysman_reading(cap / 4, cap / 2 - MiB, cap, r),
          "tolerance: below half rejected");
}

// ---- unknown device figure -------------------------------------------------

void test_unknown_global_mem()
{
    // global_mem == 0 means the caller could not determine it. Sysman's own
    // figures are then all there is; taking them beats no probe at all.
    pos2gpu::VramReading r;
    check(pos2gpu::validate_sysman_reading(9000 * MiB, 12216 * MiB, 0, r),
          "unknown cap: accepted on sysman's own figures");
    check_eq(r.free,  9000 * MiB,  "unknown cap: free unclamped");
    check_eq(r.total, 12216 * MiB, "unknown cap: sysman total used");

    // Consistency is still required.
    check(!pos2gpu::validate_sysman_reading(20000 * MiB, 12216 * MiB, 0, r),
          "unknown cap: free > total still rejected");
}

} // namespace

int main()
{
    test_arc_b580_idle();
    test_tracks_real_usage();
    test_rejects_misparse();
    test_tolerance_window();
    test_unknown_global_mem();

    if (failures) {
        std::printf("\n%d FAILURE(S)\n", failures);
        return 1;
    }
    std::printf("\nAll VRAM probe validation checks passed.\n");
    return 0;
}
