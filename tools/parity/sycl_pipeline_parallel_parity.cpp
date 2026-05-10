// sycl_pipeline_parallel_parity — exercises run_pipeline_parallel_split
// (Phase 2.1c orchestrator) and confirms the resulting fragments match
// a single-call run on the same GPU.
//
// On a single-GPU dev box this runs both halves on device 0, which
// validates the orchestrator's thread + boundary-buffer plumbing
// independent of cross-GPU peer access. Real two-device validation
// runs on the VPS via `--devices 0,1`.

#include "gpu/SyclBackend.hpp"
#include "host/GpuPipeline.hpp"
#include "host/MultiGpuPipelineParallel.hpp"
#include "host/PoolSizing.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
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

std::vector<std::uint64_t> run_full(int k, std::uint8_t seed)
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
    std::vector<std::uint64_t> out(frags.begin(), frags.end());
    sycl::free(pinned_dst, q);
    return out;
}

std::vector<std::uint64_t> run_orchestrator(
    int k, std::uint8_t seed, int dev_first, int dev_second,
    pos2gpu::PipelineStageTier tier_first  = pos2gpu::PipelineStageTier::Tiny,
    pos2gpu::PipelineStageTier tier_second = pos2gpu::PipelineStageTier::Tiny)
{
    pos2gpu::GpuPipelineConfig cfg;
    cfg.k        = k;
    cfg.strength = 2;
    derive_plot_id(cfg.plot_id, seed);

    auto r = pos2gpu::run_pipeline_parallel_split(
        cfg,
        std::vector<int>{dev_first, dev_second},
        std::vector<pos2gpu::PipelineStageTier>{tier_first, tier_second});
    auto frags = r.fragments();
    return std::vector<std::uint64_t>(frags.begin(), frags.end());
}

char const* tier_name(pos2gpu::PipelineStageTier t)
{
    return t == pos2gpu::PipelineStageTier::Tiny ? "tiny" : "min";
}

bool run_one_tiers(int k, std::uint8_t seed, int dev_first, int dev_second,
                   pos2gpu::PipelineStageTier tier_first,
                   pos2gpu::PipelineStageTier tier_second)
{
    auto ref = run_full(k, seed);
    auto pp  = run_orchestrator(k, seed, dev_first, dev_second, tier_first, tier_second);

    std::sort(ref.begin(), ref.end());
    std::sort(pp.begin(),  pp.end());

    bool const size_ok  = (ref.size() == pp.size());
    bool const bytes_ok = size_ok && std::memcmp(
        ref.data(), pp.data(),
        sizeof(std::uint64_t) * ref.size()) == 0;
    bool const ok = size_ok && bytes_ok;

    std::printf(
        "%s pipeline-parallel k=%d seed=%u dev=[%d,%d] tiers=[%s+%s] [count=%llu vs %llu size=%d bytes=%d]\n",
        ok ? "PASS" : "FAIL", k, static_cast<unsigned>(seed),
        dev_first, dev_second,
        tier_name(tier_first), tier_name(tier_second),
        static_cast<unsigned long long>(ref.size()),
        static_cast<unsigned long long>(pp.size()),
        size_ok ? 1 : 0, bytes_ok ? 1 : 0);
    return ok;
}

bool run_one(int k, std::uint8_t seed, int dev_first, int dev_second)
{
    auto ref = run_full(k, seed);
    auto pp  = run_orchestrator(k, seed, dev_first, dev_second);

    std::sort(ref.begin(), ref.end());
    std::sort(pp.begin(),  pp.end());

    bool const size_ok  = (ref.size() == pp.size());
    bool const bytes_ok = size_ok && std::memcmp(
        ref.data(), pp.data(),
        sizeof(std::uint64_t) * ref.size()) == 0;
    bool const ok = size_ok && bytes_ok;

    std::printf(
        "%s pipeline-parallel k=%d seed=%u dev=[%d,%d] [count=%llu vs %llu size=%d bytes=%d]\n",
        ok ? "PASS" : "FAIL", k, static_cast<unsigned>(seed),
        dev_first, dev_second,
        static_cast<unsigned long long>(ref.size()),
        static_cast<unsigned long long>(pp.size()),
        size_ok ? 1 : 0, bytes_ok ? 1 : 0);
    return ok;
}

} // namespace

// Run a multi-plot batch through the pipelined orchestrator and
// verify each output matches the single-call reference. Validates
// depth-N buffer-slot recycling and the inter-stage handoff.
bool run_batch(int k, int dev_first, int dev_second,
               std::size_t plot_count = 4, int depth = 2)
{
    std::vector<pos2gpu::GpuPipelineConfig> cfgs(plot_count);
    std::vector<std::vector<std::uint64_t>> refs(plot_count);
    for (std::size_t i = 0; i < cfgs.size(); ++i) {
        cfgs[i].k = k;
        cfgs[i].strength = 2;
        derive_plot_id(cfgs[i].plot_id, static_cast<std::uint8_t>(7 + i));
        refs[i] = run_full(k, static_cast<std::uint8_t>(7 + i));
        std::sort(refs[i].begin(), refs[i].end());
    }

    auto const t0 = std::chrono::steady_clock::now();
    auto batch = pos2gpu::run_pipeline_parallel_batch(
        cfgs, std::vector<int>{dev_first, dev_second}, depth);
    auto const t1 = std::chrono::steady_clock::now();
    double const wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool all_ok = true;
    for (std::size_t i = 0; i < cfgs.size(); ++i) {
        auto frags = batch[i].fragments();
        std::vector<std::uint64_t> got(frags.begin(), frags.end());
        std::sort(got.begin(), got.end());
        bool const ok = (got.size() == refs[i].size()) &&
            std::memcmp(got.data(), refs[i].data(),
                        sizeof(std::uint64_t) * got.size()) == 0;
        std::printf(
            "%s pipeline-batch k=%d entry=%zu dev=[%d,%d] [count=%llu vs %llu]\n",
            ok ? "PASS" : "FAIL", k, i, dev_first, dev_second,
            static_cast<unsigned long long>(refs[i].size()),
            static_cast<unsigned long long>(got.size()));
        if (!ok) all_ok = false;
    }
    std::printf("[wall] pipeline-batch-2stage k=%d N=%zu plots depth=%d wall=%.1fms (%.1fms/plot)\n",
                k, cfgs.size(), depth, wall_ms, wall_ms / cfgs.size());
    return all_ok;
}

// Phase 2.2c: 3-stage orchestrator parity. N=3 splits Xs+T1 / T2 /
// T3+Frag across three devices. Validates byte-identical fragments
// vs the single-GPU full-pipeline reference at k = 18, 20, 22.
bool run_one_3stage(int k, std::uint8_t seed,
                    int dev0, int dev1, int dev2)
{
    auto ref = run_full(k, seed);

    pos2gpu::GpuPipelineConfig cfg;
    cfg.k        = k;
    cfg.strength = 2;
    derive_plot_id(cfg.plot_id, seed);

    auto results = pos2gpu::run_pipeline_parallel_batch(
        std::vector<pos2gpu::GpuPipelineConfig>{cfg},
        std::vector<int>{dev0, dev1, dev2},
        /*depth=*/2);

    auto frags = results[0].fragments();
    std::vector<std::uint64_t> got(frags.begin(), frags.end());
    std::sort(ref.begin(), ref.end());
    std::sort(got.begin(), got.end());

    bool const size_ok  = (ref.size() == got.size());
    bool const bytes_ok = size_ok && std::memcmp(
        ref.data(), got.data(),
        sizeof(std::uint64_t) * ref.size()) == 0;
    bool const ok = size_ok && bytes_ok;

    std::printf(
        "%s pipeline-parallel-3stage k=%d seed=%u dev=[%d,%d,%d] "
        "[count=%llu vs %llu size=%d bytes=%d]\n",
        ok ? "PASS" : "FAIL", k, static_cast<unsigned>(seed),
        dev0, dev1, dev2,
        static_cast<unsigned long long>(ref.size()),
        static_cast<unsigned long long>(got.size()),
        size_ok ? 1 : 0, bytes_ok ? 1 : 0);
    return ok;
}

// Pipelined 3-stage batch — multiple plots in flight at each
// boundary. Validates depth-N slot recycling and outputs wall time.
bool run_batch_3stage(int k, int dev0, int dev1, int dev2,
                      std::size_t plot_count = 4, int depth = 2)
{
    std::vector<pos2gpu::GpuPipelineConfig> cfgs(plot_count);
    std::vector<std::vector<std::uint64_t>> refs(plot_count);
    for (std::size_t i = 0; i < cfgs.size(); ++i) {
        cfgs[i].k = k;
        cfgs[i].strength = 2;
        derive_plot_id(cfgs[i].plot_id, static_cast<std::uint8_t>(7 + i));
        refs[i] = run_full(k, static_cast<std::uint8_t>(7 + i));
        std::sort(refs[i].begin(), refs[i].end());
    }

    auto const t0 = std::chrono::steady_clock::now();
    auto batch = pos2gpu::run_pipeline_parallel_batch(
        cfgs, std::vector<int>{dev0, dev1, dev2}, depth);
    auto const t1 = std::chrono::steady_clock::now();
    double const wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool all_ok = true;
    for (std::size_t i = 0; i < cfgs.size(); ++i) {
        auto frags = batch[i].fragments();
        std::vector<std::uint64_t> got(frags.begin(), frags.end());
        std::sort(got.begin(), got.end());
        bool const ok = (got.size() == refs[i].size()) &&
            std::memcmp(got.data(), refs[i].data(),
                        sizeof(std::uint64_t) * got.size()) == 0;
        std::printf(
            "%s pipeline-batch-3stage k=%d entry=%zu dev=[%d,%d,%d] [count=%llu vs %llu]\n",
            ok ? "PASS" : "FAIL", k, i, dev0, dev1, dev2,
            static_cast<unsigned long long>(refs[i].size()),
            static_cast<unsigned long long>(got.size()));
        if (!ok) all_ok = false;
    }
    std::printf("[wall] pipeline-batch-3stage k=%d N=%zu plots depth=%d wall=%.1fms (%.1fms/plot)\n",
                k, cfgs.size(), depth, wall_ms, wall_ms / cfgs.size());
    return all_ok;
}

int main(int argc, char** argv)
{
    int single_k = -1;
    int dev_first = 0;
    int dev_second = 0;
    int dev_third = 0;
    bool batch_only = false;
    bool three_stage = false;
    int plot_count = 4;
    int depth = 2;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--batch") {
            batch_only = true;
        } else if (arg == "--3stage") {
            three_stage = true;
        } else if (arg == "--k" && i + 1 < argc) {
            std::from_chars(argv[i+1], argv[i+1] + std::strlen(argv[i+1]), single_k);
            ++i;
        } else if (arg == "--dev-first" && i + 1 < argc) {
            std::from_chars(argv[i+1], argv[i+1] + std::strlen(argv[i+1]), dev_first);
            ++i;
        } else if (arg == "--dev-second" && i + 1 < argc) {
            std::from_chars(argv[i+1], argv[i+1] + std::strlen(argv[i+1]), dev_second);
            ++i;
        } else if (arg == "--dev-third" && i + 1 < argc) {
            std::from_chars(argv[i+1], argv[i+1] + std::strlen(argv[i+1]), dev_third);
            ++i;
        } else if (arg == "--plots" && i + 1 < argc) {
            std::from_chars(argv[i+1], argv[i+1] + std::strlen(argv[i+1]), plot_count);
            ++i;
        } else if (arg == "--depth" && i + 1 < argc) {
            std::from_chars(argv[i+1], argv[i+1] + std::strlen(argv[i+1]), depth);
            ++i;
        }
    }

    bool all_ok = true;
    if (batch_only) {
        if (three_stage) {
            all_ok = run_batch_3stage(single_k >= 0 ? single_k : 22,
                                       dev_first, dev_second, dev_third,
                                       static_cast<std::size_t>(plot_count),
                                       depth) && all_ok;
        } else {
            all_ok = run_batch(single_k >= 0 ? single_k : 22,
                               dev_first, dev_second,
                               static_cast<std::size_t>(plot_count),
                               depth) && all_ok;
        }
    } else if (single_k >= 0) {
        all_ok = run_one(single_k, 7, dev_first, dev_second) && all_ok;
    } else {
        for (int k : {18, 20, 22}) {
            for (std::uint8_t seed : {7u, 31u}) {
                all_ok = run_one(k, seed, dev_first, dev_second) && all_ok;
            }
        }
        // Also validate the pipelined-batch path at k=22.
        all_ok = run_batch(22, dev_first, dev_second) && all_ok;

        // Phase 2-E: validate all 4 (tiny|minimal) × (tiny|minimal)
        // tier combos at the boundary handoff. Default tiny+tiny is
        // already covered above; the other 3 combos exercise the
        // h_t2_meta producer/consumer agreement post Phase 2-D fix.
        using T = pos2gpu::PipelineStageTier;
        for (auto combo : {std::pair{T::Tiny,    T::Minimal},
                           std::pair{T::Minimal, T::Tiny},
                           std::pair{T::Minimal, T::Minimal}}) {
            all_ok = run_one_tiers(20, 7, dev_first, dev_second,
                                   combo.first, combo.second) && all_ok;
        }

        // Phase 2-C: select_pipeline_devices unit tests via injected
        // VRAM lookup. Doesn't touch the GPUs — pure policy logic.
        auto check_select = [&](int a, int b,
                                std::uint64_t va, std::uint64_t vb,
                                int want_first, int want_second,
                                bool want_reordered, char const* label)
        {
            auto vram = [&](int id) -> std::uint64_t {
                if (id == a) return va;
                if (id == b) return vb;
                std::abort();
            };
            auto r = pos2gpu::select_pipeline_devices(std::vector<int>{a, b}, vram);
            bool ok = r.dev_first == want_first
                   && r.dev_second == want_second
                   && r.reordered == want_reordered
                   && r.dev_first_vram_bytes  == (want_first  == a ? va : vb)
                   && r.dev_second_vram_bytes == (want_second == a ? va : vb);
            std::printf("%s select-pipeline-devices %s: in=(%d,%d) "
                        "vram=(%llu,%llu) -> out=(%d,%d) reordered=%d\n",
                        ok ? "PASS" : "FAIL", label, a, b,
                        (unsigned long long)va, (unsigned long long)vb,
                        r.dev_first, r.dev_second, r.reordered ? 1 : 0);
            return ok;
        };
        // Already in big-VRAM-first order: no swap.
        all_ok = check_select(0, 1, 24ULL<<30, 12ULL<<30, 0, 1, false, "uniform-order") && all_ok;
        // Reverse order: swap to put big card on stage 1.
        all_ok = check_select(0, 1, 12ULL<<30, 24ULL<<30, 1, 0, true,  "swap-needed")  && all_ok;
        // Equal VRAM (uniform rig): keep input order.
        all_ok = check_select(2, 3, 16ULL<<30, 16ULL<<30, 2, 3, false, "tie-keeps-order") && all_ok;

        // Phase 2.2c: 3-stage orchestrator. N=3 splits Xs+T1 / T2 /
        // T3+Frag using the new T1-sort boundary alongside the
        // existing T2-sort boundary. Run on the same physical device
        // (dev=[0,0,0]) for the validation matrix; multi-GPU
        // configurations (dev=[0,1,2]) need to be invoked via CLI
        // args on a 3-GPU host.
        for (int k : {18, 20, 22}) {
            for (std::uint8_t seed : {7u, 31u}) {
                all_ok = run_one_3stage(k, seed, dev_first, dev_second, dev_third) && all_ok;
            }
        }
        all_ok = run_batch_3stage(22, dev_first, dev_second, dev_third) && all_ok;
    }
    return all_ok ? 0 : 1;
}
