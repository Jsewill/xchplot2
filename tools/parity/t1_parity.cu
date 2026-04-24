// t1_parity — proves the GPU T1 matching kernel produces the same SET of
// T1Pairings as pos2-chip's CPU Table1Constructor::construct.
//
// Bit-exactness comparison: both outputs are sorted lexicographically by
// (match_info, meta_lo, meta_hi) — making the comparison order-independent.
// This tests "same T1 set", which is what determines correctness for the
// downstream T2/T3/proof pipeline.

#include "gpu/AesGpu.cuh"
#include "gpu/XsKernel.cuh"
#include "gpu/T1Kernel.cuh"

// pos2-chip headers for the CPU reference.
#include "plot/PlotLayout.hpp"
#include "plot/TableConstructorGeneric.hpp"
#include "pos/ProofCore.hpp"
#include "pos/ProofParams.hpp"

#include "ParityCommon.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using pos2gpu::parity::derive_plot_id;

#define CHECK(call) do {                                                                     \
    cudaError_t err = (call);                                                                \
    if (err != cudaSuccess) {                                                                \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                   \
                     cudaGetErrorString(err));                                               \
        std::exit(2);                                                                        \
    }                                                                                        \
} while (0)

struct PairKey {
    uint32_t mi;  // match_info
    uint32_t lo;  // meta_lo
    uint32_t hi;  // meta_hi
    bool operator<(PairKey const& o) const noexcept {
        if (mi != o.mi) return mi < o.mi;
        if (hi != o.hi) return hi < o.hi;
        return lo < o.lo;
    }
    bool operator==(PairKey const& o) const noexcept {
        return mi == o.mi && lo == o.lo && hi == o.hi;
    }
};

bool run_for_id(std::array<uint8_t, 32> const& plot_id, char const* label, int k, int strength)
{
    uint64_t const total = 1ULL << k;

    std::printf("[%s  k=%d  strength=%d  N=%llu]\n",
                label, k, strength, static_cast<unsigned long long>(total));

    ProofParams params(plot_id.data(),
                       static_cast<uint8_t>(k),
                       static_cast<uint8_t>(strength),
                       /*testnet=*/uint8_t{0});

    // ---- CPU reference (XsConstructor → Table1Constructor::construct) ----
    size_t max_section_pairs = max_pairs_per_section_possible(params);
    size_t num_sections      = static_cast<size_t>(params.get_num_sections());
    size_t max_pairs         = max_section_pairs * num_sections;
    size_t max_element_bytes = std::max({sizeof(Xs_Candidate), sizeof(T1Pairing),
                                         sizeof(T2Pairing), sizeof(T3Pairing)});
    PlotLayout layout(max_section_pairs, num_sections, max_element_bytes,
                      /*minor_scratch_bytes=*/2 * 1024 * 1024);

    auto xsV = layout.xs();
    XsConstructor xs_ctor(params);
    auto xs_sorted = xs_ctor.construct(xsV.out, xsV.post_sort_tmp, xsV.minor);

    // Mirror Plotter::run() — if XsConstructor returned its output in the
    // PrimaryOut slot, copy it aside so T1's construct (which writes its
    // output into PrimaryOut) doesn't corrupt the input mid-construction.
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
    pos2gpu::XsCandidateGpu* d_xs = nullptr;
    CHECK(cudaMalloc(&d_xs, sizeof(pos2gpu::XsCandidateGpu) * total));
    size_t xs_temp_bytes = 0;
    CHECK(pos2gpu::launch_construct_xs(plot_id.data(), k, false, nullptr, nullptr, &xs_temp_bytes));
    void* d_xs_temp = nullptr;
    CHECK(cudaMalloc(&d_xs_temp, xs_temp_bytes));
    CHECK(pos2gpu::launch_construct_xs(plot_id.data(), k, false, d_xs, d_xs_temp, &xs_temp_bytes));
    CHECK(cudaDeviceSynchronize());

    auto t1p = pos2gpu::make_t1_params(k, strength);
    // Capacity: max_pairs from PlotLayout is the CPU's worst-case sizing;
    // re-use it.
    uint64_t capacity = static_cast<uint64_t>(max_pairs);

    // T1 match emits SoA: (uint64 meta, uint32 mi) parallel streams.
    uint64_t* d_t1_meta = nullptr;
    uint32_t* d_t1_mi   = nullptr;
    CHECK(cudaMalloc(&d_t1_meta, sizeof(uint64_t) * capacity));
    CHECK(cudaMalloc(&d_t1_mi,   sizeof(uint32_t) * capacity));
    uint64_t* d_t1_count = nullptr;
    CHECK(cudaMalloc(&d_t1_count, sizeof(uint64_t)));

    size_t t1_temp_bytes = 0;
    CHECK(pos2gpu::launch_t1_match(plot_id.data(), t1p, d_xs, total,
                                   nullptr, nullptr, d_t1_count, capacity,
                                   nullptr, &t1_temp_bytes));
    void* d_t1_temp = nullptr;
    CHECK(cudaMalloc(&d_t1_temp, t1_temp_bytes));
    CHECK(pos2gpu::launch_t1_match(plot_id.data(), t1p, d_xs, total,
                                   d_t1_meta, d_t1_mi, d_t1_count, capacity,
                                   d_t1_temp, &t1_temp_bytes));
    CHECK(cudaDeviceSynchronize());

    uint64_t gpu_count = 0;
    CHECK(cudaMemcpy(&gpu_count, d_t1_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    auto free_all = [&]() {
        cudaFree(d_t1_temp); cudaFree(d_t1_count);
        cudaFree(d_t1_meta); cudaFree(d_t1_mi);
        cudaFree(d_xs_temp); cudaFree(d_xs);
    };

    if (gpu_count > capacity) {
        std::printf("  GPU OVERFLOW: emitted %llu but capacity %llu\n",
                    (unsigned long long)gpu_count, (unsigned long long)capacity);
        free_all();
        return false;
    }

    std::vector<uint64_t> h_meta(gpu_count);
    std::vector<uint32_t> h_mi  (gpu_count);
    if (gpu_count > 0) {
        CHECK(cudaMemcpy(h_meta.data(), d_t1_meta, sizeof(uint64_t) * gpu_count, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_mi.data(),   d_t1_mi,   sizeof(uint32_t) * gpu_count, cudaMemcpyDeviceToHost));
    }
    free_all();

    std::vector<PairKey> gpu_keys;
    gpu_keys.reserve(gpu_count);
    for (uint64_t i = 0; i < gpu_count; ++i) {
        uint32_t meta_lo = uint32_t(h_meta[i]);
        uint32_t meta_hi = uint32_t(h_meta[i] >> 32);
        gpu_keys.push_back({h_mi[i], meta_lo, meta_hi});
    }
    std::sort(gpu_keys.begin(), gpu_keys.end());

    std::printf("  GPU produced %zu T1Pairings\n", gpu_keys.size());

    // ---- compare ----
    if (cpu_keys.size() != gpu_keys.size()) {
        std::printf("  count mismatch (CPU %zu vs GPU %zu) — analysing overlap\n",
                    cpu_keys.size(), gpu_keys.size());

        // Build a set of CPU pairs (by sorted vector) and binary-search GPU.
        size_t in_cpu_only = 0, in_gpu_only = 0, common = 0;
        std::vector<PairKey> only_in_gpu;
        size_t i = 0, j = 0;
        while (i < cpu_keys.size() && j < gpu_keys.size()) {
            if (cpu_keys[i] == gpu_keys[j]) { ++common; ++i; ++j; }
            else if (cpu_keys[i] < gpu_keys[j]) { ++in_cpu_only; ++i; }
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
            uint32_t k_const = static_cast<uint32_t>(k);
            uint32_t x_l = (p.lo >> 0); // meta_lo holds bottom 32 bits of (x_l<<k)|x_r
            uint32_t x_r = 0;
            // meta = (x_l << k) | x_r, in 36 bits for k=18. Reconstruct:
            uint64_t meta = (uint64_t(p.hi) << 32) | uint64_t(p.lo);
            x_l = static_cast<uint32_t>(meta >> k_const);
            x_r = static_cast<uint32_t>(meta & ((1ULL << k_const) - 1));
            std::printf("    GPU-only sample: x_l=%u x_r=%u  match_info=0x%05x\n",
                        x_l, x_r, p.mi);
        }
        return false;
    }

    uint64_t mismatches = 0;
    for (size_t i = 0; i < cpu_keys.size(); ++i) {
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
    } else {
        std::printf("  FAIL  %llu mismatches / %zu\n",
                    (unsigned long long)mismatches, cpu_keys.size());
        return false;
    }
}

} // namespace

int main()
{
    pos2gpu::initialize_aes_tables();

    bool all_ok = true;
    // Fixed plot_id for historical reproducibility.
    {
        std::array<uint8_t, 32> id{};
        for (auto& b : id) b = 0xab;
        all_ok = run_for_id(id, "plot_id=0xab*32", /*k=*/18, /*strength=*/2) && all_ok;
    }
    // Wide seed coverage at strength=2, k=18.
    for (uint32_t seed : {
            // primes
            1u, 2u, 3u, 5u, 7u, 11u, 13u, 17u,
            19u, 23u, 29u, 31u, 42u, 1337u,
            0xCAFEBABEu, 0xDEADBEEFu,
            // boundary seeds
            0u, 0xFFFFFFFFu, 0x80000000u,
         }) {
        all_ok = run_for_id(derive_plot_id(seed),
                            (std::string("seed=") + std::to_string(seed)).c_str(),
                            /*k=*/18, /*strength=*/2) && all_ok;
    }
    // Strength sweep at k=18. strength=7 leaves num_match_target_bits=9,
    // still above the FINE_BITS=8 floor in the match kernels.
    for (int strength : {3, 4, 5, 6, 7}) {
        for (uint32_t seed : {1u, 17u}) {
            char label[64];
            std::snprintf(label, sizeof(label), "seed=%u strength=%d", seed, strength);
            all_ok = run_for_id(derive_plot_id(seed), label,
                                /*k=*/18, strength) && all_ok;
        }
    }

    std::printf("\n==> %s\n", all_ok ? "ALL OK" : "FAIL");
    return all_ok ? 0 : 1;
}
