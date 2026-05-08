// sycl_phase_split_parity — exercises the Phase 2.1a/b first-half +
// second-half entry points and confirms the resulting plot fragments
// match a single-call run on the same GPU.

#include "gpu/SyclBackend.hpp"
#include "host/GpuPipeline.hpp"
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
    } else {
        for (int k : {18, 20, 22}) {
            for (std::uint8_t seed : {7u, 31u}) {
                all_ok = run_one(k, seed) && all_ok;
            }
        }
    }
    return all_ok ? 0 : 1;
}
