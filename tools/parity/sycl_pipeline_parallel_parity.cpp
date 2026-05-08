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
    int k, std::uint8_t seed, int dev_first, int dev_second)
{
    pos2gpu::GpuPipelineConfig cfg;
    cfg.k        = k;
    cfg.strength = 2;
    derive_plot_id(cfg.plot_id, seed);

    auto r = pos2gpu::run_pipeline_parallel_split(cfg, dev_first, dev_second);
    auto frags = r.fragments();
    return std::vector<std::uint64_t>(frags.begin(), frags.end());
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

int main(int argc, char** argv)
{
    int single_k = -1;
    int dev_first = 0;
    int dev_second = 0;
    for (int i = 1; i + 1 < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--k") {
            std::from_chars(argv[i+1], argv[i+1] + std::strlen(argv[i+1]), single_k);
        } else if (arg == "--dev-first") {
            std::from_chars(argv[i+1], argv[i+1] + std::strlen(argv[i+1]), dev_first);
        } else if (arg == "--dev-second") {
            std::from_chars(argv[i+1], argv[i+1] + std::strlen(argv[i+1]), dev_second);
        }
    }

    bool all_ok = true;
    if (single_k >= 0) {
        all_ok = run_one(single_k, 7, dev_first, dev_second) && all_ok;
    } else {
        for (int k : {18, 20, 22}) {
            for (std::uint8_t seed : {7u, 31u}) {
                all_ok = run_one(k, seed, dev_first, dev_second) && all_ok;
            }
        }
    }
    return all_ok ? 0 : 1;
}
