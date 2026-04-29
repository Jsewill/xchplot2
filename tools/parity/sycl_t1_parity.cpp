// sycl_t1_parity — SYCL-native sibling of t1_parity.cu. Builds on every
// backend (CUDA / HIP / Level Zero / OMP) so the T1 matcher can be
// validated against the pos2-chip CPU reference on AMD and Intel
// devices, where the .cu version isn't compiled.
//
// Same comparison semantics as t1_parity.cu: both CPU and GPU outputs
// are sorted by (match_info, meta_hi, meta_lo) and compared as a set.
// Bit-exactness of the SET is what determines correctness for the
// downstream T2/T3/proof pipeline — the post-construct sort by
// match_info collapses the order in which matches were emitted.
//
// Usage:
//   ./sycl_t1_parity                       # default sweep
//   ./sycl_t1_parity --k 20                # single-k smoke test
//   ./sycl_t1_parity --k 20 --strength 4   # custom strength
//
// The default sweep stays small (k <= 18) so it fits on 8 GiB cards
// and so the CPU reference completes in seconds. --k lets a triage
// session push the matcher to the largest k that fits on the device.

#include "gpu/AesGpu.cuh"
#include "gpu/SyclBackend.hpp"
#include "gpu/XsKernel.cuh"
#include "gpu/T1Kernel.cuh"

#include "plot/PlotLayout.hpp"
#include "plot/TableConstructorGeneric.hpp"
#include "pos/ProofCore.hpp"
#include "pos/ProofParams.hpp"

#include "ParityCommon.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {

using pos2gpu::parity::derive_plot_id;

struct PairKey {
    uint32_t mi;
    uint32_t lo;
    uint32_t hi;
    bool operator<(PairKey const& o) const noexcept {
        if (mi != o.mi) return mi < o.mi;
        if (hi != o.hi) return hi < o.hi;
        return lo < o.lo;
    }
    bool operator==(PairKey const& o) const noexcept {
        return mi == o.mi && lo == o.lo && hi == o.hi;
    }
};

template <typename T>
T* sycl_alloc_device(sycl::queue& q, std::size_t n, char const* what)
{
    T* p = sycl::malloc_device<T>(n, q);
    if (!p) {
        std::fprintf(stderr, "  FAIL: sycl::malloc_device(%s, %zu * %zu B)\n",
                     what, n, sizeof(T));
        std::exit(2);
    }
    return p;
}

bool run_for_id(sycl::queue& q,
                std::array<uint8_t, 32> const& plot_id,
                char const* label,
                int k,
                int strength)
{
    uint64_t const total = 1ULL << k;
    std::printf("[%s  k=%d  strength=%d  N=%llu]\n",
                label, k, strength, static_cast<unsigned long long>(total));

    ProofParams params(plot_id.data(),
                       static_cast<uint8_t>(k),
                       static_cast<uint8_t>(strength),
                       /*testnet=*/uint8_t{0});

    // ---- CPU reference (XsConstructor → Table1Constructor::construct) ----
    std::size_t max_section_pairs = max_pairs_per_section_possible(params);
    std::size_t num_sections      = static_cast<std::size_t>(params.get_num_sections());
    std::size_t max_pairs         = max_section_pairs * num_sections;
    std::size_t max_element_bytes = std::max({sizeof(Xs_Candidate), sizeof(T1Pairing),
                                              sizeof(T2Pairing), sizeof(T3Pairing)});
    PlotLayout layout(max_section_pairs, num_sections, max_element_bytes,
                      /*minor_scratch_bytes=*/2 * 1024 * 1024);

    auto xsV = layout.xs();
    XsConstructor xs_ctor(params);
    auto xs_sorted = xs_ctor.construct(xsV.out, xsV.post_sort_tmp, xsV.minor);

    // Mirror t1_parity.cu: if XsConstructor returned its output in the
    // PrimaryOut slot, copy aside so T1's construct (which writes its
    // output into PrimaryOut) doesn't corrupt the input.
    if (xs_sorted.data() == xsV.out.data()) {
        std::copy(xsV.out.begin(), xsV.out.end(), xsV.post_sort_tmp.begin());
        xs_sorted = xsV.post_sort_tmp.first(xs_sorted.size());
    }

    auto t1V = layout.t1();
    Table1Constructor t1_ctor(params, t1V.target, t1V.minor);
    auto t1_pairs = t1_ctor.construct(xs_sorted, t1V.out, t1V.post_sort_tmp);

    std::vector<PairKey> cpu_keys;
    cpu_keys.reserve(t1_pairs.size());
    for (auto const& p : t1_pairs) {
        cpu_keys.push_back({p.match_info, p.meta_lo, p.meta_hi});
    }
    std::sort(cpu_keys.begin(), cpu_keys.end());
    std::printf("  CPU produced %zu T1Pairings\n", cpu_keys.size());

    // ---- GPU pipeline: launch_construct_xs, then launch_t1_match ----
    auto* d_xs = sycl_alloc_device<pos2gpu::XsCandidateGpu>(q, total, "d_xs");

    std::size_t xs_temp_bytes = 0;
    pos2gpu::launch_construct_xs(plot_id.data(), k, /*testnet=*/false,
                                 nullptr, nullptr, &xs_temp_bytes, q);
    void* d_xs_temp = sycl_alloc_device<unsigned char>(q, xs_temp_bytes, "d_xs_temp");
    pos2gpu::launch_construct_xs(plot_id.data(), k, /*testnet=*/false,
                                 d_xs, d_xs_temp, &xs_temp_bytes, q);
    q.wait();

    auto t1p = pos2gpu::make_t1_params(k, strength);
    uint64_t const capacity = static_cast<uint64_t>(max_pairs);

    auto* d_t1_meta  = sycl_alloc_device<uint64_t>(q, capacity, "d_t1_meta");
    auto* d_t1_mi    = sycl_alloc_device<uint32_t>(q, capacity, "d_t1_mi");
    auto* d_t1_count = sycl_alloc_device<uint64_t>(q, 1,        "d_t1_count");

    // Mirror GpuPipeline.cpp: the streaming pipeline always memsets
    // d_counter to 0 before the real launch_t1_match call. The size-
    // query call below doesn't touch d_t1_count, but the real call's
    // launch_t1_match_prepare also memsets it — keep the explicit
    // pre-zero to make the test a one-shot if the prepare path ever
    // changes.
    q.memset(d_t1_count, 0, sizeof(uint64_t)).wait();

    std::size_t t1_temp_bytes = 0;
    pos2gpu::launch_t1_match(plot_id.data(), t1p, d_xs, total,
                             nullptr, nullptr, d_t1_count, capacity,
                             nullptr, &t1_temp_bytes, q);
    void* d_t1_temp = sycl_alloc_device<unsigned char>(q, t1_temp_bytes, "d_t1_temp");
    pos2gpu::launch_t1_match(plot_id.data(), t1p, d_xs, total,
                             d_t1_meta, d_t1_mi, d_t1_count, capacity,
                             d_t1_temp, &t1_temp_bytes, q);
    q.wait();

    uint64_t gpu_count = 0;
    q.memcpy(&gpu_count, d_t1_count, sizeof(uint64_t)).wait();

    auto free_all = [&]() {
        sycl::free(d_t1_temp,  q);
        sycl::free(d_t1_count, q);
        sycl::free(d_t1_mi,    q);
        sycl::free(d_t1_meta,  q);
        sycl::free(d_xs_temp,  q);
        sycl::free(d_xs,       q);
    };

    if (gpu_count > capacity) {
        std::printf("  GPU OVERFLOW: emitted %llu but capacity %llu\n",
                    static_cast<unsigned long long>(gpu_count),
                    static_cast<unsigned long long>(capacity));
        free_all();
        return false;
    }

    std::vector<uint64_t> h_meta(gpu_count);
    std::vector<uint32_t> h_mi  (gpu_count);
    if (gpu_count > 0) {
        q.memcpy(h_meta.data(), d_t1_meta, sizeof(uint64_t) * gpu_count).wait();
        q.memcpy(h_mi.data(),   d_t1_mi,   sizeof(uint32_t) * gpu_count).wait();
    }
    free_all();

    std::vector<PairKey> gpu_keys;
    gpu_keys.reserve(gpu_count);
    for (uint64_t i = 0; i < gpu_count; ++i) {
        uint32_t meta_lo = static_cast<uint32_t>(h_meta[i]);
        uint32_t meta_hi = static_cast<uint32_t>(h_meta[i] >> 32);
        gpu_keys.push_back({h_mi[i], meta_lo, meta_hi});
    }
    std::sort(gpu_keys.begin(), gpu_keys.end());
    std::printf("  GPU produced %zu T1Pairings\n", gpu_keys.size());

    if (cpu_keys.size() != gpu_keys.size()) {
        std::printf("  count mismatch (CPU %zu vs GPU %zu) — analysing overlap\n",
                    cpu_keys.size(), gpu_keys.size());
        std::size_t in_cpu_only = 0, in_gpu_only = 0, common = 0;
        std::vector<PairKey> only_in_gpu;
        std::size_t i = 0, j = 0;
        while (i < cpu_keys.size() && j < gpu_keys.size()) {
            if (cpu_keys[i] == gpu_keys[j])      { ++common; ++i; ++j; }
            else if (cpu_keys[i] < gpu_keys[j])  { ++in_cpu_only; ++i; }
            else {
                if (only_in_gpu.size() < 5) only_in_gpu.push_back(gpu_keys[j]);
                ++in_gpu_only; ++j;
            }
        }
        in_cpu_only += cpu_keys.size() - i;
        while (j < gpu_keys.size()) {
            if (only_in_gpu.size() < 5) only_in_gpu.push_back(gpu_keys[j]);
            ++in_gpu_only;
            ++j;
        }
        std::printf("    common=%zu  cpu_only=%zu  gpu_only=%zu\n",
                    common, in_cpu_only, in_gpu_only);
        for (auto const& p : only_in_gpu) {
            uint64_t meta = (uint64_t(p.hi) << 32) | uint64_t(p.lo);
            uint32_t x_l  = static_cast<uint32_t>(meta >> static_cast<uint32_t>(k));
            uint32_t x_r  = static_cast<uint32_t>(meta & ((1ULL << k) - 1));
            std::printf("    GPU-only sample: x_l=%u x_r=%u  match_info=0x%08x\n",
                        x_l, x_r, p.mi);
        }
        return false;
    }

    uint64_t mismatches = 0;
    for (std::size_t i = 0; i < cpu_keys.size(); ++i) {
        if (!(cpu_keys[i] == gpu_keys[i])) {
            if (mismatches < 5) {
                std::printf("  MISMATCH at i=%zu  cpu=(mi=0x%08x lo=0x%08x hi=0x%08x)  "
                            "gpu=(mi=0x%08x lo=0x%08x hi=0x%08x)\n",
                            i,
                            cpu_keys[i].mi, cpu_keys[i].lo, cpu_keys[i].hi,
                            gpu_keys[i].mi, gpu_keys[i].lo, gpu_keys[i].hi);
            }
            ++mismatches;
        }
    }
    if (mismatches == 0) {
        std::printf("  OK  %zu / %zu T1Pairings match (sorted set comparison)\n",
                    cpu_keys.size(), cpu_keys.size());
        return true;
    }
    std::printf("  FAIL  %llu mismatches / %zu\n",
                static_cast<unsigned long long>(mismatches), cpu_keys.size());
    return false;
}

bool parse_int_arg(std::string_view sv, int& out)
{
    auto const* first = sv.data();
    auto const* last  = sv.data() + sv.size();
    auto r = std::from_chars(first, last, out);
    return r.ec == std::errc{} && r.ptr == last;
}

} // namespace

int main(int argc, char** argv)
{
    pos2gpu::initialize_aes_tables();

    int k_override        = -1;
    int strength_override = -1;
    for (int i = 1; i + 1 < argc; ++i) {
        std::string_view a = argv[i];
        if      (a == "--k")        { (void)parse_int_arg(argv[++i], k_override); }
        else if (a == "--strength") { (void)parse_int_arg(argv[++i], strength_override); }
    }

    sycl::queue q{ sycl::gpu_selector_v };
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    bool all_ok = true;

    if (k_override > 0) {
        int const s = (strength_override > 0) ? strength_override : 2;
        // Use the same fixed plot_id family as the default sweep so a
        // user-driven --k 22 run is reproducible alongside the seed=1
        // baseline.
        std::string label = "k=" + std::to_string(k_override) +
                            " strength=" + std::to_string(s);
        all_ok = run_for_id(q, derive_plot_id(/*seed=*/1u),
                            label.c_str(), k_override, s) && all_ok;
    } else {
        // Default sweep — k=18 only, since launch_t1_match_prepare rejects
        // k < 18 (smallest size for which num_match_target_bits exceeds the
        // FINE_BITS=8 floor with sensible margin). Seed and strength
        // coverage is deliberately narrower than t1_parity.cu because
        // this binary is meant to be run as a quick-triage check on
        // AMD/Intel hardware where the CUDA test isn't available — the
        // full coverage is in t1_parity.cu on the CUDA build path.
        for (uint32_t seed : { 1u, 7u, 31u, 0xCAFEBABEu, 0xDEADBEEFu }) {
            std::string label = "seed=" + std::to_string(seed);
            all_ok = run_for_id(q, derive_plot_id(seed),
                                label.c_str(), /*k=*/18, /*strength=*/2)
                     && all_ok;
        }
        // Strength sweep at k=18 — exercises the test_mask path through
        // the matcher which scales with strength. strength=7 leaves
        // num_match_target_bits=9, still above the FINE_BITS=8 floor.
        for (int strength : { 3, 4, 5, 6, 7 }) {
            std::string label = "seed=1 strength=" + std::to_string(strength);
            all_ok = run_for_id(q, derive_plot_id(1u),
                                label.c_str(), /*k=*/18, strength)
                     && all_ok;
        }
    }

    std::printf("\n==> %s\n", all_ok ? "ALL OK" : "FAIL");
    return all_ok ? 0 : 1;
}
