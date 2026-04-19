// t3_parity — proves the GPU T3 final-table kernel produces the same SET
// of T3Pairings as pos2-chip's CPU Table3Constructor::construct.
//
// Feeds the SAME sorted T2Pairing[] to both CPU and GPU T3 to isolate T3
// from upstream phases (already validated by t1_parity / t2_parity).

#include "gpu/AesGpu.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T3Kernel.cuh"

#include "plot/PlotLayout.hpp"
#include "plot/TableConstructorGeneric.hpp"
#include "pos/ProofConstants.hpp"
#include "pos/ProofCore.hpp"
#include "pos/ProofParams.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

#define CHECK(call) do {                                                                     \
    cudaError_t err = (call);                                                                \
    if (err != cudaSuccess) {                                                                \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                   \
                     cudaGetErrorString(err));                                               \
        std::exit(2);                                                                        \
    }                                                                                        \
} while (0)

std::array<uint8_t, 32> derive_plot_id(uint32_t seed)
{
    std::array<uint8_t, 32> id{};
    uint64_t s = 0x9E3779B97F4A7C15ULL ^ uint64_t(seed) * 0x100000001B3ULL;
    for (size_t i = 0; i < id.size(); ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        id[i] = static_cast<uint8_t>(s >> 56);
    }
    return id;
}

bool run_for_id(std::array<uint8_t, 32> const& plot_id, char const* label, int k, int strength)
{
    uint64_t const total = 1ULL << k;
    std::printf("[%s  k=%d  strength=%d  N=%llu]\n",
                label, k, strength, static_cast<unsigned long long>(total));

    ProofParams params(plot_id.data(),
                       static_cast<uint8_t>(k),
                       static_cast<uint8_t>(strength),
                       /*testnet=*/uint8_t{0});

    // ---- CPU pipeline: Xs → T1 → T2 → T3 ----
    size_t max_section_pairs = max_pairs_per_section_possible(params);
    size_t num_sections      = static_cast<size_t>(params.get_num_sections());
    size_t max_pairs         = max_section_pairs * num_sections;
    size_t max_element_bytes = std::max({sizeof(Xs_Candidate), sizeof(T1Pairing),
                                         sizeof(T2Pairing), sizeof(T3Pairing)});
    PlotLayout layout(max_section_pairs, num_sections, max_element_bytes,
                      /*minor_scratch_bytes=*/2 * 1024 * 1024);

    // Xs
    auto xsV = layout.xs();
    XsConstructor xs_ctor(params);
    auto xs_sorted = xs_ctor.construct(xsV.out, xsV.post_sort_tmp, xsV.minor);
    if (xs_sorted.data() == xsV.out.data()) {
        std::copy(xsV.out.begin(), xsV.out.end(), xsV.post_sort_tmp.begin());
        xs_sorted = xsV.post_sort_tmp.first(xs_sorted.size());
    }

    // T1
    auto t1V = layout.t1();
    Table1Constructor t1_ctor(params, t1V.target, t1V.minor);
    auto t1_pairs = t1_ctor.construct(xs_sorted, t1V.out, t1V.post_sort_tmp);
    if (t1_pairs.data() == t1V.out.data()) {
        std::copy(t1V.out.begin(), t1V.out.begin() + t1_pairs.size(),
                  t1V.post_sort_tmp.begin());
        t1_pairs = t1V.post_sort_tmp.first(t1_pairs.size());
    }

    // T2
    auto t2V = layout.t2();
    Table2Constructor t2_ctor(params, t2V.target, t2V.minor);
    auto t2_pairs = t2_ctor.construct(t1_pairs, t2V.out, t2V.post_sort_tmp);
    if (t2_pairs.data() == t2V.out.data()) {
        std::copy(t2V.out.begin(), t2V.out.begin() + t2_pairs.size(),
                  t2V.post_sort_tmp.begin());
        t2_pairs = t2V.post_sort_tmp.first(t2_pairs.size());
    }
    std::printf("  CPU T2 produced %zu T2Pairings\n", t2_pairs.size());

    // Snapshot T2 — needed for the GPU T3 input AND survives the CPU T3
    // construct call which reuses memory.
    std::vector<T2Pairing> t2_snapshot(t2_pairs.begin(), t2_pairs.end());

    // T3 (CPU)
    auto t3V = layout.t3();
    Table3Constructor t3_ctor(params, t3V.target, t3V.minor);
    auto t3_pairs = t3_ctor.construct(t2_pairs, t3V.out, t3V.post_sort_tmp);
    std::printf("  CPU T3 produced %zu T3Pairings\n", t3_pairs.size());

    std::vector<uint64_t> cpu_fragments;
    cpu_fragments.reserve(t3_pairs.size());
    for (auto const& p : t3_pairs) cpu_fragments.push_back(p.proof_fragment);
    std::sort(cpu_fragments.begin(), cpu_fragments.end());

    // ---- GPU T3 — feed it the CPU's snapshot of sorted T2 ----
    static_assert(sizeof(T2Pairing) == sizeof(pos2gpu::T2PairingGpu),
                  "T2Pairing layout drift");

    // T3 input is SoA: separate meta / xbits / match_info streams.
    std::vector<uint64_t> h_t2_meta(t2_snapshot.size());
    std::vector<uint32_t> h_t2_xbits(t2_snapshot.size());
    std::vector<uint32_t> h_t2_mi(t2_snapshot.size());
    for (size_t i = 0; i < t2_snapshot.size(); ++i) {
        h_t2_meta[i]  = t2_snapshot[i].meta;
        h_t2_xbits[i] = t2_snapshot[i].x_bits;
        h_t2_mi[i]    = t2_snapshot[i].match_info;
    }
    uint64_t* d_t2_meta = nullptr;
    uint32_t* d_t2_xbits = nullptr;
    uint32_t* d_t2_mi = nullptr;
    CHECK(cudaMalloc(&d_t2_meta,  sizeof(uint64_t) * h_t2_meta.size()));
    CHECK(cudaMalloc(&d_t2_xbits, sizeof(uint32_t) * h_t2_xbits.size()));
    CHECK(cudaMalloc(&d_t2_mi,    sizeof(uint32_t) * h_t2_mi.size()));
    CHECK(cudaMemcpy(d_t2_meta,  h_t2_meta.data(),
                     sizeof(uint64_t) * h_t2_meta.size(),  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2_xbits, h_t2_xbits.data(),
                     sizeof(uint32_t) * h_t2_xbits.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2_mi,    h_t2_mi.data(),
                     sizeof(uint32_t) * h_t2_mi.size(),    cudaMemcpyHostToDevice));

    auto t3p = pos2gpu::make_t3_params(k, strength);
    uint64_t capacity = static_cast<uint64_t>(max_pairs);

    pos2gpu::T3PairingGpu* d_t3 = nullptr;
    CHECK(cudaMalloc(&d_t3, sizeof(pos2gpu::T3PairingGpu) * capacity));
    uint64_t* d_t3_count = nullptr;
    CHECK(cudaMalloc(&d_t3_count, sizeof(uint64_t)));

    size_t t3_temp_bytes = 0;
    CHECK(pos2gpu::launch_t3_match(plot_id.data(), t3p,
                                   d_t2_meta, d_t2_xbits, nullptr,
                                   t2_snapshot.size(),
                                   d_t3, d_t3_count, capacity,
                                   nullptr, &t3_temp_bytes));
    void* d_t3_temp = nullptr;
    CHECK(cudaMalloc(&d_t3_temp, t3_temp_bytes));
    CHECK(pos2gpu::launch_t3_match(plot_id.data(), t3p,
                                   d_t2_meta, d_t2_xbits, d_t2_mi,
                                   t2_snapshot.size(),
                                   d_t3, d_t3_count, capacity,
                                   d_t3_temp, &t3_temp_bytes));
    CHECK(cudaDeviceSynchronize());

    uint64_t gpu_count = 0;
    CHECK(cudaMemcpy(&gpu_count, d_t3_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    if (gpu_count > capacity) {
        std::printf("  GPU OVERFLOW: %llu / %llu\n",
                    (unsigned long long)gpu_count, (unsigned long long)capacity);
        cudaFree(d_t3_temp); cudaFree(d_t3_count); cudaFree(d_t3);
        cudaFree(d_t2_mi); cudaFree(d_t2_xbits); cudaFree(d_t2_meta);
        return false;
    }

    std::vector<pos2gpu::T3PairingGpu> gpu_pairs(gpu_count);
    if (gpu_count > 0) {
        CHECK(cudaMemcpy(gpu_pairs.data(), d_t3,
                         sizeof(pos2gpu::T3PairingGpu) * gpu_count,
                         cudaMemcpyDeviceToHost));
    }
    cudaFree(d_t3_temp); cudaFree(d_t3_count); cudaFree(d_t3);
        cudaFree(d_t2_mi); cudaFree(d_t2_xbits); cudaFree(d_t2_meta);

    std::vector<uint64_t> gpu_fragments;
    gpu_fragments.reserve(gpu_pairs.size());
    for (auto const& p : gpu_pairs) gpu_fragments.push_back(p.proof_fragment);
    std::sort(gpu_fragments.begin(), gpu_fragments.end());

    std::printf("  GPU T3 produced %zu T3Pairings\n", gpu_fragments.size());

    if (cpu_fragments.size() != gpu_fragments.size()) {
        std::printf("  FAIL: count mismatch (CPU %zu vs GPU %zu)\n",
                    cpu_fragments.size(), gpu_fragments.size());
        return false;
    }

    uint64_t mismatches = 0;
    for (size_t i = 0; i < cpu_fragments.size(); ++i) {
        if (cpu_fragments[i] != gpu_fragments[i]) {
            if (mismatches < 5) {
                std::printf("  MISMATCH at i=%zu  cpu=0x%016llx  gpu=0x%016llx\n",
                            i,
                            (unsigned long long)cpu_fragments[i],
                            (unsigned long long)gpu_fragments[i]);
            }
            ++mismatches;
        }
    }

    if (mismatches == 0) {
        std::printf("  OK  %zu / %zu T3 fragments match\n",
                    cpu_fragments.size(), cpu_fragments.size());
        return true;
    } else {
        std::printf("  FAIL  %llu mismatches / %zu\n",
                    (unsigned long long)mismatches, cpu_fragments.size());
        return false;
    }
}

} // namespace

int main()
{
    pos2gpu::initialize_aes_tables();

    bool all_ok = true;
    {
        std::array<uint8_t, 32> id{};
        for (auto& b : id) b = 0xab;
        all_ok = run_for_id(id, "plot_id=0xab*32", /*k=*/18, /*strength=*/2) && all_ok;
    }
    for (uint32_t seed : {
            1u, 2u, 3u, 5u, 7u, 11u, 13u, 17u,
            19u, 23u, 29u, 31u, 42u, 1337u,
            0xCAFEBABEu, 0xDEADBEEFu,
         }) {
        all_ok = run_for_id(derive_plot_id(seed),
                            (std::string("seed=") + std::to_string(seed)).c_str(),
                            /*k=*/18, /*strength=*/2) && all_ok;
    }
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
