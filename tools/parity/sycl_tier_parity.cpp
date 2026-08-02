// sycl_tier_parity — every streaming tier must produce the SAME plot.
//
// WHY THIS EXISTS
// ---------------
// The tiers (plain / compact / minimal / tiny) are four different routes to
// one answer: they trade VRAM for host RAM and PCIe traffic, but the fragment
// stream they emit is defined to be byte-identical. Nothing in the suite
// checked that. Every other parity test validates a KERNEL or a PRIMITIVE in
// isolation, so a tier-specific defect in the surrounding host orchestration
// — the slicing, the parking, the per-pass tile staging — had no oracle at
// all, and each tier stayed happily self-consistent while being wrong.
//
// That is not hypothetical. Tiny's T1 sub-section match staged its tile as
// two USM copies and waited only the second; when the un-waited half lost the
// race, one of the 16 T1 passes matched exactly zero, and the plot came out
// 78% of the right size and completely wrong. T1/T2/T3 counts stayed
// internally consistent, the file was structurally valid, no kernel was at
// fault, and the entire parity suite passed. Comparing tiny against plain
// would have flagged it instantly.
//
// Scope, stated honestly: this catches DETERMINISTIC tier divergence. The
// defect above was a race, so this test would only have caught it on the runs
// where it fired (~8%). The always-on zero-yield guard in the T1 match is what
// catches that one reliably; this is the complementary net for the whole
// class of tier-specific regressions that are deterministic — a mis-sliced
// range, an off-by-one park, a tile boundary that drops entries.
//
// Runs at a small k by default so it is cheap enough to sit in the routine
// suite. k is the first argument; the tiny sub-section path (4 sections x
// num_match_keys passes) engages at every k this supports, so a small k
// exercises the same code shape as k=28.
//
//   sycl_tier_parity [k] [strength]

#include "gpu/SyclBackend.hpp"
#include "host/GpuPipeline.hpp"
#include "host/PoolSizing.hpp"

#include <sycl/sycl.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct TierSpec {
    char const* name;
    bool        plain_mode;
    bool        tiny_mode;
    int         t2_tile_count;
    int         gather_tile_count;
    bool        spill;   // route every table this tier can spill to disk
};

// Mirrors BatchPlotter's tier configuration (BatchPlotter.cpp ~2298). Keep in
// step with it — if the two drift, this test silently stops covering a tier
// that ships.
//
// The trailing spilled variants are the host-RAM disk-offload path. They are
// not tiers the picker can choose; they are the same tier with its cold tables
// redirected to a TempFile, which is exactly the kind of change that stays
// invisible until a plot is farmed. Byte parity against plain is the whole
// contract, so it belongs here rather than in an ad-hoc run: a spill that
// drops or reorders entries produces a valid-looking plot, same as the T1
// race did.
constexpr TierSpec kTiers[] = {
    {"plain",       true,  false, 2, 1, false},
    {"compact",     false, false, 2, 1, false},
    {"minimal",     false, false, 8, 4, false},
    {"tiny",        false, true,  8, 4, false},
    {"minimal+dsk", false, false, 8, 4, true},
    {"tiny+disk",   false, true,  8, 4, true},
};

void derive_plot_id(std::array<uint8_t, 32>& out, uint8_t seed)
{
    for (int i = 0; i < 32; ++i) {
        out[i] = static_cast<uint8_t>(seed * 17u + i * 19u);
    }
}

struct Run {
    std::vector<uint64_t> fragments;
    uint64_t t1 = 0, t2 = 0, t3 = 0;
    bool     ok = false;
};

Run run_tier(TierSpec const& spec, int k, int strength, uint8_t seed)
{
    Run r;
    pos2gpu::GpuPipelineConfig cfg;
    derive_plot_id(cfg.plot_id, seed);
    cfg.k        = k;
    cfg.strength = strength;

    auto& q = pos2gpu::sycl_backend::queue();
    int const num_section_bits = (k < 28) ? 2 : (k - 26);
    uint64_t const cap =
        pos2gpu::max_pairs_per_section(k, num_section_bits) *
        (1ULL << num_section_bits);

    // The h_* scratch pointers stay null: the pipeline then allocates them
    // per-plot. Slower than the batch path, but it keeps this test free of
    // pool plumbing, and it is the same shape the one-shot `test` mode uses.
    uint64_t* pinned_dst = static_cast<uint64_t*>(
        sycl::malloc_host(cap * sizeof(uint64_t), q));
    if (!pinned_dst) {
        std::printf("FAIL  tier=%-8s could not allocate %llu-entry pinned dst\n",
                    spec.name, (unsigned long long)cap);
        return r;
    }

    pos2gpu::StreamingPinnedScratch scratch{};
    scratch.plain_mode        = spec.plain_mode;
    scratch.tiny_mode         = spec.tiny_mode;
    scratch.t2_tile_count     = spec.t2_tile_count;
    scratch.gather_tile_count = spec.gather_tile_count;
    if (spec.spill) {
        // Mirrors BatchPlotter's routable set (BatchPlotter.cpp ~2156), which
        // is tier-dependent: h_t1_meta and h_t2_meta exist as separate buffers
        // only in Tiny, and h_frags is Compact/Minimal only. Setting a bit the
        // tier cannot honour is not a no-op — it nulls a buffer nothing will
        // redirect — so gate them the same way the policy does.
        scratch.spill.h_t3       = true;
        scratch.spill.h_t2_xbits = true;
        scratch.spill.h_t1_meta  = spec.tiny_mode;
        scratch.spill.h_t2_meta  = spec.tiny_mode;
        scratch.spill.h_frags    = !spec.tiny_mode;
    }

    try {
        auto res  = pos2gpu::run_gpu_pipeline_streaming(cfg, pinned_dst, cap, scratch);
        auto span = res.fragments();
        // Copy out before freeing pinned_dst — the result BORROWS it.
        r.fragments.assign(span.begin(), span.end());
        r.t1 = res.t1_count;
        r.t2 = res.t2_count;
        r.t3 = res.t3_count;
        r.ok = true;
    } catch (std::exception const& e) {
        std::printf("FAIL  tier=%-8s threw: %s\n", spec.name, e.what());
    }
    sycl::free(pinned_dst, q);
    return r;
}

}  // namespace

int main(int argc, char** argv)
{
    int const k        = (argc > 1) ? std::atoi(argv[1]) : 22;
    int const strength = (argc > 2) ? std::atoi(argv[2]) : 2;
    uint8_t const seed = 7;

    std::printf("=== tier parity: k=%d strength=%d ===\n", k, strength);

    Run reference;
    char const* ref_name = nullptr;
    int failures = 0;

    for (auto const& spec : kTiers) {
        Run r = run_tier(spec, k, strength, seed);
        if (!r.ok) { ++failures; continue; }

        if (!ref_name) {
            reference = std::move(r);
            ref_name  = spec.name;
            std::printf("REF   tier=%-8s T1=%llu T2=%llu T3=%llu  frags=%zu\n",
                        spec.name,
                        (unsigned long long)reference.t1,
                        (unsigned long long)reference.t2,
                        (unsigned long long)reference.t3,
                        reference.fragments.size());
            continue;
        }

        // Counts first: they localise a divergence to a phase, which a raw
        // fragment mismatch does not. A short T1 with consistent downstream
        // counts is the signature of a whole unit of work going missing.
        bool const counts_ok = (r.t1 == reference.t1) &&
                               (r.t2 == reference.t2) &&
                               (r.t3 == reference.t3);
        bool const size_ok   = r.fragments.size() == reference.fragments.size();
        bool const bytes_ok  = size_ok &&
            std::memcmp(r.fragments.data(), reference.fragments.data(),
                        r.fragments.size() * sizeof(uint64_t)) == 0;

        bool const ok = counts_ok && bytes_ok;
        if (!ok) ++failures;
        std::printf("%s  tier=%-8s T1=%llu T2=%llu T3=%llu  frags=%zu"
                    "  [counts=%d bytes=%d]\n",
                    ok ? "PASS" : "FAIL", spec.name,
                    (unsigned long long)r.t1, (unsigned long long)r.t2,
                    (unsigned long long)r.t3, r.fragments.size(),
                    counts_ok, bytes_ok);

        if (!counts_ok) {
            // Report the ratio: an exact rational (15/16, and its square and
            // fourth power downstream) means whole passes vanished rather
            // than data being scrambled, and names the granularity.
            std::printf("      vs %s: T1 %llu/%llu = %.6f, T2 %.6f, T3 %.6f\n",
                        ref_name,
                        (unsigned long long)r.t1, (unsigned long long)reference.t1,
                        reference.t1 ? double(r.t1) / double(reference.t1) : 0.0,
                        reference.t2 ? double(r.t2) / double(reference.t2) : 0.0,
                        reference.t3 ? double(r.t3) / double(reference.t3) : 0.0);
        }
    }

    if (!ref_name) {
        std::printf("\nFAIL: no tier completed — nothing to compare\n");
        return 1;
    }
    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nall tiers agree\n", failures);
    return failures ? 1 : 0;
}
