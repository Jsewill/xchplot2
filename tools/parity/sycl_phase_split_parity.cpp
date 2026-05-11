// sycl_phase_split_parity — exercises the Phase 2.1a/b first-half +
// second-half entry points and confirms the resulting plot fragments
// match a single-call run on the same GPU.

#include "gpu/SyclBackend.hpp"
#include "host/GpuPipeline.hpp"
#include "host/HostPinnedPool.hpp"
#include "host/PoolSizing.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

namespace {

void derive_plot_id(std::array<uint8_t, 32>& out, std::uint8_t seed)
{
    for (int i = 0; i < 32; ++i) {
        out[i] = static_cast<uint8_t>(seed * 17u + i * 19u);
    }
}

struct RunResult {
    std::vector<std::uint64_t> fragments;
    std::uint64_t              t2_count = 0;
};

RunResult run_full(int k, std::uint8_t seed)
{
    pos2gpu::GpuPipelineConfig cfg;
    cfg.k        = k;
    cfg.strength = 2;
    derive_plot_id(cfg.plot_id, seed);

    auto& q = pos2gpu::sycl_backend::queue();
    int const num_section_bits = (k < 28) ? 2 : (k - 26);
    std::uint64_t const cap =
        pos2gpu::max_pairs_per_section(k, num_section_bits) *
        (1ULL << num_section_bits);

    std::uint64_t* pinned_dst = static_cast<std::uint64_t*>(
        sycl::malloc_host(cap * sizeof(std::uint64_t), q));
    if (!pinned_dst) std::exit(2);

    pos2gpu::StreamingPinnedScratch scratch{};
    scratch.tiny_mode         = true;
    scratch.t2_tile_count     = 8;
    scratch.gather_tile_count = 4;
    auto r = pos2gpu::run_gpu_pipeline_streaming(cfg, pinned_dst, cap, scratch);

    auto frags = r.fragments();
    RunResult out;
    out.fragments.assign(frags.begin(), frags.end());
    out.t2_count = r.t2_count;
    sycl::free(pinned_dst, q);
    return out;
}

RunResult run_split(int k, std::uint8_t seed)
{
    pos2gpu::GpuPipelineConfig cfg;
    cfg.k        = k;
    cfg.strength = 2;
    derive_plot_id(cfg.plot_id, seed);

    auto& q = pos2gpu::sycl_backend::queue();
    int const num_section_bits = (k < 28) ? 2 : (k - 26);
    std::uint64_t const cap =
        pos2gpu::max_pairs_per_section(k, num_section_bits) *
        (1ULL << num_section_bits);

    // Distinct buffers for T1 (h_meta) and T2 (h_t2_meta) — the
    // buffer-reuse fix relies on these NOT aliasing in tiny mode.
    std::uint64_t* h_meta = static_cast<std::uint64_t*>(
        sycl::malloc_host(cap * sizeof(std::uint64_t), q));
    std::uint64_t* h_t2_meta = static_cast<std::uint64_t*>(
        sycl::malloc_host(cap * sizeof(std::uint64_t), q));
    std::uint32_t* h_t2_xbits = static_cast<std::uint32_t*>(
        sycl::malloc_host(cap * sizeof(std::uint32_t), q));
    std::uint32_t* h_keys_merged = static_cast<std::uint32_t*>(
        sycl::malloc_host(cap * sizeof(std::uint32_t), q));
    std::uint64_t* pinned_dst = static_cast<std::uint64_t*>(
        sycl::malloc_host(cap * sizeof(std::uint64_t), q));
    if (!h_meta || !h_t2_meta || !h_t2_xbits || !h_keys_merged || !pinned_dst) {
        std::exit(2);
    }

    pos2gpu::StreamingPinnedScratch first{};
    first.tiny_mode          = true;
    first.t2_tile_count      = 8;
    first.gather_tile_count  = 4;
    first.h_meta             = h_meta;
    first.h_t2_meta          = h_t2_meta;
    first.h_t2_xbits         = h_t2_xbits;
    first.h_keys_merged      = h_keys_merged;
    first.stop_after_t2_sort = true;
    auto r1 = pos2gpu::run_gpu_pipeline_streaming(cfg, pinned_dst, cap, first);

    pos2gpu::StreamingPinnedScratch second{};
    second.tiny_mode         = true;
    second.t2_tile_count     = 8;
    second.gather_tile_count = 4;
    second.h_meta            = h_meta;
    second.h_t2_meta         = h_t2_meta;
    second.h_t2_xbits        = h_t2_xbits;
    second.h_keys_merged     = h_keys_merged;
    second.start_at_t3_match = true;
    second.t1_count_in       = r1.t1_count;
    second.t2_count_in       = r1.t2_count;
    auto r2 = pos2gpu::run_gpu_pipeline_streaming(cfg, pinned_dst, cap, second);

    auto frags = r2.fragments();
    RunResult out;
    out.fragments.assign(frags.begin(), frags.end());
    out.t2_count = r1.t2_count;

    sycl::free(h_meta,        q);
    sycl::free(h_t2_meta,     q);
    sycl::free(h_t2_xbits,    q);
    sycl::free(h_keys_merged, q);
    sycl::free(pinned_dst,    q);
    return out;
}

bool run_one(int k, std::uint8_t seed)
{
    auto ref   = run_full(k, seed);
    auto split = run_split(k, seed);

    std::sort(ref.fragments.begin(),   ref.fragments.end());
    std::sort(split.fragments.begin(), split.fragments.end());

    bool const t2_match = (ref.t2_count == split.t2_count);
    bool const size_ok  = (ref.fragments.size() == split.fragments.size());
    bool const bytes_ok = size_ok && std::memcmp(
        ref.fragments.data(), split.fragments.data(),
        sizeof(std::uint64_t) * ref.fragments.size()) == 0;
    bool const ok = t2_match && size_ok && bytes_ok;

    std::printf(
        "%s phase-split k=%d seed=%u  t2[ref=%llu split=%llu]  t3[ref=%llu split=%llu] size=%d bytes=%d\n",
        ok ? "PASS" : "FAIL", k, static_cast<unsigned>(seed),
        static_cast<unsigned long long>(ref.t2_count),
        static_cast<unsigned long long>(split.t2_count),
        static_cast<unsigned long long>(ref.fragments.size()),
        static_cast<unsigned long long>(split.fragments.size()),
        size_ok ? 1 : 0, bytes_ok ? 1 : 0);
    return ok;
}

// Phase 2.2 T1-sort split: producer runs Xs/T1/T1-sort and exits with
// stop_after_t1_sort; consumer runs T2 match through end with
// start_at_t2_match, reading sorted T1 from caller-provided h_meta +
// h_keys_merged. End-to-end fragments must match the full reference.
RunResult run_t1_split(int k, std::uint8_t seed)
{
    pos2gpu::GpuPipelineConfig cfg;
    cfg.k        = k;
    cfg.strength = 2;
    derive_plot_id(cfg.plot_id, seed);

    auto& q = pos2gpu::sycl_backend::queue();
    int const num_section_bits = (k < 28) ? 2 : (k - 26);
    std::uint64_t const cap =
        pos2gpu::max_pairs_per_section(k, num_section_bits) *
        (1ULL << num_section_bits);

    // Both halves use h_meta for sorted T1 metadata and h_keys_merged
    // for sorted T1 match_info. h_t2_meta / h_t2_xbits are needed only
    // by the consumer's T2 match output (filled inside, not by us).
    std::uint64_t* h_meta = static_cast<std::uint64_t*>(
        sycl::malloc_host(cap * sizeof(std::uint64_t), q));
    std::uint64_t* h_t2_meta = static_cast<std::uint64_t*>(
        sycl::malloc_host(cap * sizeof(std::uint64_t), q));
    std::uint32_t* h_t2_xbits = static_cast<std::uint32_t*>(
        sycl::malloc_host(cap * sizeof(std::uint32_t), q));
    std::uint32_t* h_keys_merged = static_cast<std::uint32_t*>(
        sycl::malloc_host(cap * sizeof(std::uint32_t), q));
    std::uint64_t* pinned_dst = static_cast<std::uint64_t*>(
        sycl::malloc_host(cap * sizeof(std::uint64_t), q));
    if (!h_meta || !h_t2_meta || !h_t2_xbits || !h_keys_merged || !pinned_dst) {
        std::exit(2);
    }

    pos2gpu::StreamingPinnedScratch first{};
    first.tiny_mode          = true;
    first.t2_tile_count      = 8;
    first.gather_tile_count  = 4;
    first.h_meta             = h_meta;
    first.h_t2_meta          = h_t2_meta;
    first.h_t2_xbits         = h_t2_xbits;
    first.h_keys_merged      = h_keys_merged;
    first.stop_after_t1_sort = true;
    auto r1 = pos2gpu::run_gpu_pipeline_streaming(cfg, pinned_dst, cap, first);

    pos2gpu::StreamingPinnedScratch second{};
    second.tiny_mode         = true;
    second.t2_tile_count     = 8;
    second.gather_tile_count = 4;
    second.h_meta            = h_meta;
    second.h_t2_meta         = h_t2_meta;
    second.h_t2_xbits        = h_t2_xbits;
    second.h_keys_merged     = h_keys_merged;
    second.start_at_t2_match = true;
    second.t1_count_in       = r1.t1_count;
    auto r2 = pos2gpu::run_gpu_pipeline_streaming(cfg, pinned_dst, cap, second);

    auto frags = r2.fragments();
    RunResult out;
    out.fragments.assign(frags.begin(), frags.end());
    out.t2_count = r2.t2_count;

    sycl::free(h_meta,        q);
    sycl::free(h_t2_meta,     q);
    sycl::free(h_t2_xbits,    q);
    sycl::free(h_keys_merged, q);
    sycl::free(pinned_dst,    q);
    return out;
}

bool run_one_t1_split(int k, std::uint8_t seed)
{
    auto ref   = run_full(k, seed);
    auto split = run_t1_split(k, seed);

    std::sort(ref.fragments.begin(),   ref.fragments.end());
    std::sort(split.fragments.begin(), split.fragments.end());

    bool const size_ok  = (ref.fragments.size() == split.fragments.size());
    bool const bytes_ok = size_ok && std::memcmp(
        ref.fragments.data(), split.fragments.data(),
        sizeof(std::uint64_t) * ref.fragments.size()) == 0;
    bool const ok = size_ok && bytes_ok;

    std::printf(
        "%s phase-split-t1 k=%d seed=%u  t3[ref=%llu split=%llu] size=%d bytes=%d\n",
        ok ? "PASS" : "FAIL", k, static_cast<unsigned>(seed),
        static_cast<unsigned long long>(ref.fragments.size()),
        static_cast<unsigned long long>(split.fragments.size()),
        size_ok ? 1 : 0, bytes_ok ? 1 : 0);
    return ok;
}

// Phase 2.2f: T1-sort split with minimal tier on either or both
// halves. Validates that the new minimal-mode start_at_t2_match
// rehydration (d_t1_meta_sorted + d_t1_keys_merged H2D) and the
// stop_after_t1_sort hand-off in non-tiny mode work end-to-end.
RunResult run_t1_split_tiers(int k, std::uint8_t seed,
                             bool first_tiny, bool second_tiny)
{
    pos2gpu::GpuPipelineConfig cfg;
    cfg.k        = k;
    cfg.strength = 2;
    derive_plot_id(cfg.plot_id, seed);

    auto& q = pos2gpu::sycl_backend::queue();
    int const num_section_bits = (k < 28) ? 2 : (k - 26);
    std::uint64_t const cap =
        pos2gpu::max_pairs_per_section(k, num_section_bits) *
        (1ULL << num_section_bits);

    auto* h_meta        = static_cast<std::uint64_t*>(sycl::malloc_host(cap * sizeof(std::uint64_t), q));
    auto* h_t2_meta     = static_cast<std::uint64_t*>(sycl::malloc_host(cap * sizeof(std::uint64_t), q));
    auto* h_t2_xbits    = static_cast<std::uint32_t*>(sycl::malloc_host(cap * sizeof(std::uint32_t), q));
    auto* h_keys_merged = static_cast<std::uint32_t*>(sycl::malloc_host(cap * sizeof(std::uint32_t), q));
    auto* pinned_dst    = static_cast<std::uint64_t*>(sycl::malloc_host(cap * sizeof(std::uint64_t), q));
    if (!h_meta || !h_t2_meta || !h_t2_xbits || !h_keys_merged || !pinned_dst) std::exit(2);

    pos2gpu::StreamingPinnedScratch first{};
    first.tiny_mode          = first_tiny;
    first.t2_tile_count      = 8;
    first.gather_tile_count  = 4;
    first.h_meta             = h_meta;
    first.h_t2_meta          = h_t2_meta;
    first.h_t2_xbits         = h_t2_xbits;
    first.h_keys_merged      = h_keys_merged;
    first.stop_after_t1_sort = true;
    auto r1 = pos2gpu::run_gpu_pipeline_streaming(cfg, pinned_dst, cap, first);

    pos2gpu::StreamingPinnedScratch second{};
    second.tiny_mode         = second_tiny;
    second.t2_tile_count     = 8;
    second.gather_tile_count = 4;
    second.h_meta            = h_meta;
    second.h_t2_meta         = h_t2_meta;
    second.h_t2_xbits        = h_t2_xbits;
    second.h_keys_merged     = h_keys_merged;
    second.start_at_t2_match = true;
    second.t1_count_in       = r1.t1_count;
    auto r2 = pos2gpu::run_gpu_pipeline_streaming(cfg, pinned_dst, cap, second);

    auto frags = r2.fragments();
    RunResult out;
    out.fragments.assign(frags.begin(), frags.end());
    out.t2_count = r2.t2_count;

    sycl::free(h_meta,        q);
    sycl::free(h_t2_meta,     q);
    sycl::free(h_t2_xbits,    q);
    sycl::free(h_keys_merged, q);
    sycl::free(pinned_dst,    q);
    return out;
}

bool run_one_t1_split_tiers(int k, std::uint8_t seed,
                            bool first_tiny, bool second_tiny)
{
    auto ref   = run_full(k, seed);
    auto split = run_t1_split_tiers(k, seed, first_tiny, second_tiny);

    std::sort(ref.fragments.begin(),   ref.fragments.end());
    std::sort(split.fragments.begin(), split.fragments.end());

    bool const size_ok  = (ref.fragments.size() == split.fragments.size());
    bool const bytes_ok = size_ok && std::memcmp(
        ref.fragments.data(), split.fragments.data(),
        sizeof(std::uint64_t) * ref.fragments.size()) == 0;
    bool const ok = size_ok && bytes_ok;

    std::printf(
        "%s phase-split-t1-tiers k=%d seed=%u tiers=[%s+%s] [t3=%llu vs %llu size=%d bytes=%d]\n",
        ok ? "PASS" : "FAIL", k, static_cast<unsigned>(seed),
        first_tiny ? "tiny" : "min", second_tiny ? "tiny" : "min",
        static_cast<unsigned long long>(ref.fragments.size()),
        static_cast<unsigned long long>(split.fragments.size()),
        size_ok ? 1 : 0, bytes_ok ? 1 : 0);
    return ok;
}

// Phase 2-D probe: invoke run_split with tiny_mode=false on the
// producer half. Validates that the GpuPipeline h_t2_meta fix lets
// minimal-mode producers populate the boundary buffers in the format
// start_at_t3_match expects, matching tiny-mode producer output.
RunResult run_split_minimal_producer(int k, std::uint8_t seed)
{
    pos2gpu::GpuPipelineConfig cfg;
    cfg.k        = k;
    cfg.strength = 2;
    derive_plot_id(cfg.plot_id, seed);

    auto& q = pos2gpu::sycl_backend::queue();
    int const num_section_bits = (k < 28) ? 2 : (k - 26);
    std::uint64_t const cap =
        pos2gpu::max_pairs_per_section(k, num_section_bits) *
        (1ULL << num_section_bits);

    auto* h_meta        = static_cast<std::uint64_t*>(sycl::malloc_host(cap * sizeof(std::uint64_t), q));
    auto* h_t2_meta     = static_cast<std::uint64_t*>(sycl::malloc_host(cap * sizeof(std::uint64_t), q));
    auto* h_t2_xbits    = static_cast<std::uint32_t*>(sycl::malloc_host(cap * sizeof(std::uint32_t), q));
    auto* h_keys_merged = static_cast<std::uint32_t*>(sycl::malloc_host(cap * sizeof(std::uint32_t), q));
    auto* pinned_dst    = static_cast<std::uint64_t*>(sycl::malloc_host(cap * sizeof(std::uint64_t), q));
    if (!h_meta || !h_t2_meta || !h_t2_xbits || !h_keys_merged || !pinned_dst) std::exit(2);

    pos2gpu::StreamingPinnedScratch first{};
    first.tiny_mode          = false;  // minimal-mode producer
    first.t2_tile_count      = 8;
    first.gather_tile_count  = 4;
    first.h_meta             = h_meta;
    first.h_t2_meta          = h_t2_meta;
    first.h_t2_xbits         = h_t2_xbits;
    first.h_keys_merged      = h_keys_merged;
    first.stop_after_t2_sort = true;
    auto r1 = pos2gpu::run_gpu_pipeline_streaming(cfg, pinned_dst, cap, first);

    pos2gpu::StreamingPinnedScratch second{};
    second.tiny_mode         = true;  // tiny-mode consumer
    second.t2_tile_count     = 8;
    second.gather_tile_count = 4;
    second.h_meta            = h_meta;
    second.h_t2_meta         = h_t2_meta;
    second.h_t2_xbits        = h_t2_xbits;
    second.h_keys_merged     = h_keys_merged;
    second.start_at_t3_match = true;
    second.t1_count_in       = r1.t1_count;
    second.t2_count_in       = r1.t2_count;
    auto r2 = pos2gpu::run_gpu_pipeline_streaming(cfg, pinned_dst, cap, second);

    auto frags = r2.fragments();
    RunResult out;
    out.fragments.assign(frags.begin(), frags.end());
    out.t2_count = r1.t2_count;

    sycl::free(h_meta,        q);
    sycl::free(h_t2_meta,     q);
    sycl::free(h_t2_xbits,    q);
    sycl::free(h_keys_merged, q);
    sycl::free(pinned_dst,    q);
    return out;
}

// HostPinnedPool integration test: run 2 plots through the same pool
// and verify byte-identical output to the no-pool reference. Pool is
// shared across plots so the second plot's h_t1_mi alloc is amortised
// (the slot is sized correctly on plot 1 and reused on plot 2).
bool run_pool_two_plots(int k, std::uint8_t seed_a, std::uint8_t seed_b)
{
    pos2gpu::HostPinnedPool pool;
    auto run_one_through_pool = [&](std::uint8_t seed) {
        pos2gpu::GpuPipelineConfig cfg;
        cfg.k        = k;
        cfg.strength = 2;
        derive_plot_id(cfg.plot_id, seed);

        auto& q = pos2gpu::sycl_backend::queue();
        int const num_section_bits = (k < 28) ? 2 : (k - 26);
        std::uint64_t const cap =
            pos2gpu::max_pairs_per_section(k, num_section_bits) *
            (1ULL << num_section_bits);
        auto* pinned_dst = static_cast<std::uint64_t*>(
            sycl::malloc_host(cap * sizeof(std::uint64_t), q));
        if (!pinned_dst) std::exit(2);

        pos2gpu::StreamingPinnedScratch s{};
        s.tiny_mode         = true;
        s.t2_tile_count     = 8;
        s.gather_tile_count = 4;
        s.pool              = &pool;
        auto r = pos2gpu::run_gpu_pipeline_streaming(cfg, pinned_dst, cap, s);
        auto frags = r.fragments();
        std::vector<std::uint64_t> out(frags.begin(), frags.end());
        sycl::free(pinned_dst, q);
        return out;
    };

    auto ref_a  = run_full(k, seed_a);
    auto ref_b  = run_full(k, seed_b);
    auto pool_a = run_one_through_pool(seed_a);
    std::size_t const slots_after_a = pool.slot_count();
    auto pool_b = run_one_through_pool(seed_b);
    std::size_t const slots_after_b = pool.slot_count();

    std::sort(ref_a.fragments.begin(),  ref_a.fragments.end());
    std::sort(ref_b.fragments.begin(),  ref_b.fragments.end());
    std::sort(pool_a.begin(),           pool_a.end());
    std::sort(pool_b.begin(),           pool_b.end());

    bool const a_match = ref_a.fragments.size() == pool_a.size() &&
        std::memcmp(ref_a.fragments.data(), pool_a.data(),
                    sizeof(std::uint64_t) * pool_a.size()) == 0;
    bool const b_match = ref_b.fragments.size() == pool_b.size() &&
        std::memcmp(ref_b.fragments.data(), pool_b.data(),
                    sizeof(std::uint64_t) * pool_b.size()) == 0;
    // Pool should keep its slots stable between plots — plot 2 must
    // reuse the slots created by plot 1, not allocate new ones.
    bool const slots_stable = (slots_after_a > 0) && (slots_after_a == slots_after_b);

    bool const ok = a_match && b_match && slots_stable;
    std::printf(
        "%s pool-two-plots k=%d seeds=[%u,%u] a_bytes=%d b_bytes=%d slots=%zu->%zu\n",
        ok ? "PASS" : "FAIL", k,
        static_cast<unsigned>(seed_a), static_cast<unsigned>(seed_b),
        a_match ? 1 : 0, b_match ? 1 : 0, slots_after_a, slots_after_b);
    return ok;
}

bool run_one_minimal_producer(int k, std::uint8_t seed)
{
    auto ref   = run_full(k, seed);
    auto split = run_split_minimal_producer(k, seed);

    std::sort(ref.fragments.begin(),   ref.fragments.end());
    std::sort(split.fragments.begin(), split.fragments.end());

    bool const t2_match = (ref.t2_count == split.t2_count);
    bool const size_ok  = (ref.fragments.size() == split.fragments.size());
    bool const bytes_ok = size_ok && std::memcmp(
        ref.fragments.data(), split.fragments.data(),
        sizeof(std::uint64_t) * ref.fragments.size()) == 0;
    bool const ok = t2_match && size_ok && bytes_ok;

    std::printf(
        "%s phase-split-min-producer k=%d seed=%u  t2[ref=%llu split=%llu]  t3[ref=%llu split=%llu] size=%d bytes=%d\n",
        ok ? "PASS" : "FAIL", k, static_cast<unsigned>(seed),
        static_cast<unsigned long long>(ref.t2_count),
        static_cast<unsigned long long>(split.t2_count),
        static_cast<unsigned long long>(ref.fragments.size()),
        static_cast<unsigned long long>(split.fragments.size()),
        size_ok ? 1 : 0, bytes_ok ? 1 : 0);
    return ok;
}

} // namespace

int main(int argc, char** argv)
{
    int single_k = -1;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string_view(argv[i]) == "--k") {
            std::from_chars(argv[i+1], argv[i+1] + std::strlen(argv[i+1]), single_k);
        }
    }

    bool all_ok = true;
    if (single_k >= 0) {
        all_ok = run_one(single_k, 7) && all_ok;
        all_ok = run_one_minimal_producer(single_k, 7) && all_ok;
        all_ok = run_pool_two_plots(single_k, 7, 31) && all_ok;
        all_ok = run_one_t1_split(single_k, 7) && all_ok;
        all_ok = run_one_t1_split_tiers(single_k, 7, false, false) && all_ok;
    } else {
        for (int k : {18, 20, 22}) {
            for (std::uint8_t seed : {7u, 31u}) {
                all_ok = run_one(k, seed) && all_ok;
                all_ok = run_one_minimal_producer(k, seed) && all_ok;
                all_ok = run_one_t1_split(k, seed) && all_ok;
            }
            all_ok = run_pool_two_plots(k, 7, 31) && all_ok;
        }
        // Phase 2.2f: T1-sort split with all 4 tier combos at the
        // boundary handoff. (tiny+tiny is already covered above by
        // run_one_t1_split — repeated here only via the other 3.)
        for (auto combo : {std::pair{false, false},   // min+min
                           std::pair{true,  false},   // tiny+min
                           std::pair{false, true}}) { // min+tiny
            all_ok = run_one_t1_split_tiers(20, 7, combo.first, combo.second) && all_ok;
        }
    }
    return all_ok ? 0 : 1;
}
