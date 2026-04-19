// t2_parity — proves the GPU T2 matching kernel produces the same SET of
// T2Pairings as pos2-chip's CPU Table2Constructor::construct.
//
// We feed the SAME sorted T1Pairing[] (produced by CPU Table1Constructor)
// to both the CPU and GPU T2 implementations to isolate T2 from T1's
// correctness, which is already validated by t1_parity.

#include "gpu/AesGpu.cuh"
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"

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

// Sort key for T2Pairing: (match_info, x_bits, meta) — fully canonicalises
// the pair regardless of emission order.
struct T2Key {
    uint32_t mi;
    uint32_t xb;
    uint64_t meta;
    bool operator<(T2Key const& o) const noexcept {
        if (mi   != o.mi)   return mi   < o.mi;
        if (xb   != o.xb)   return xb   < o.xb;
        return meta < o.meta;
    }
    bool operator==(T2Key const& o) const noexcept {
        return mi == o.mi && xb == o.xb && meta == o.meta;
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

    // ---- CPU pipeline: Xs → T1 → T2 ----
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
    // Same PrimaryOut reuse: T2 will write to t2V.out which aliases t1V.out.
    if (t1_pairs.data() == t1V.out.data()) {
        std::copy(t1V.out.begin(), t1V.out.begin() + t1_pairs.size(),
                  t1V.post_sort_tmp.begin());
        t1_pairs = t1V.post_sort_tmp.first(t1_pairs.size());
    }
    std::printf("  CPU T1 produced %zu T1Pairings\n", t1_pairs.size());

    // Snapshot the CPU's sorted T1 — we need it later for the GPU T2 input,
    // and the upcoming Table2Constructor::construct call will reuse the
    // same memory blocks.
    std::vector<T1Pairing> t1_snapshot(t1_pairs.begin(), t1_pairs.end());

    // T2 (CPU) — uses the CPU's t1_pairs as input
    auto t2V = layout.t2();
    Table2Constructor t2_ctor(params, t2V.target, t2V.minor);
    auto t2_pairs = t2_ctor.construct(t1_pairs, t2V.out, t2V.post_sort_tmp);
    std::printf("  CPU T2 produced %zu T2Pairings\n", t2_pairs.size());

    std::vector<T2Key> cpu_keys;
    cpu_keys.reserve(t2_pairs.size());
    for (auto const& p : t2_pairs) {
        cpu_keys.push_back({p.match_info, p.x_bits, p.meta});
    }
    std::sort(cpu_keys.begin(), cpu_keys.end());

    // ---- GPU T2 — feed it the CPU's snapshot of sorted T1 ----
    static_assert(sizeof(T1Pairing) == sizeof(pos2gpu::T1PairingGpu),
                  "T1Pairing layout drift");

    pos2gpu::T1PairingGpu* d_t1 = nullptr;
    CHECK(cudaMalloc(&d_t1, sizeof(pos2gpu::T1PairingGpu) * t1_snapshot.size()));
    (void)d_t1;  // no longer passed into the match kernel

    // SoA sorted-T1: separate meta (uint64) and match_info (uint32) streams.
    std::vector<uint64_t> h_t1_meta(t1_snapshot.size());
    std::vector<uint32_t> h_t1_mi  (t1_snapshot.size());
    for (size_t i = 0; i < t1_snapshot.size(); ++i) {
        h_t1_meta[i] = (uint64_t(t1_snapshot[i].meta_hi) << 32)
                     |  uint64_t(t1_snapshot[i].meta_lo);
        h_t1_mi[i]   = t1_snapshot[i].match_info;
    }
    uint64_t* d_t1_meta = nullptr;
    uint32_t* d_t1_mi   = nullptr;
    CHECK(cudaMalloc(&d_t1_meta, sizeof(uint64_t) * h_t1_meta.size()));
    CHECK(cudaMalloc(&d_t1_mi,   sizeof(uint32_t) * h_t1_mi.size()));
    CHECK(cudaMemcpy(d_t1_meta, h_t1_meta.data(),
                     sizeof(uint64_t) * h_t1_meta.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t1_mi,   h_t1_mi.data(),
                     sizeof(uint32_t) * h_t1_mi.size(),   cudaMemcpyHostToDevice));

    auto t2p = pos2gpu::make_t2_params(k, strength);
    uint64_t capacity = static_cast<uint64_t>(max_pairs);

    pos2gpu::T2PairingGpu* d_t2 = nullptr;
    CHECK(cudaMalloc(&d_t2, sizeof(pos2gpu::T2PairingGpu) * capacity));
    uint64_t* d_t2_count = nullptr;
    CHECK(cudaMalloc(&d_t2_count, sizeof(uint64_t)));

    size_t t2_temp_bytes = 0;
    CHECK(pos2gpu::launch_t2_match(plot_id.data(), t2p, nullptr, nullptr, t1_snapshot.size(),
                                   d_t2, d_t2_count, capacity,
                                   nullptr, &t2_temp_bytes));
    void* d_t2_temp = nullptr;
    CHECK(cudaMalloc(&d_t2_temp, t2_temp_bytes));
    CHECK(pos2gpu::launch_t2_match(plot_id.data(), t2p, d_t1_meta, d_t1_mi, t1_snapshot.size(),
                                   d_t2, d_t2_count, capacity,
                                   d_t2_temp, &t2_temp_bytes));
    CHECK(cudaDeviceSynchronize());

    uint64_t gpu_count = 0;
    CHECK(cudaMemcpy(&gpu_count, d_t2_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    if (gpu_count > capacity) {
        std::printf("  GPU OVERFLOW: %llu / %llu\n",
                    (unsigned long long)gpu_count, (unsigned long long)capacity);
        cudaFree(d_t2_temp); cudaFree(d_t2_count); cudaFree(d_t2); cudaFree(d_t1_mi); cudaFree(d_t1_meta); cudaFree(d_t1);
        return false;
    }

    std::vector<pos2gpu::T2PairingGpu> gpu_pairs(gpu_count);
    if (gpu_count > 0) {
        CHECK(cudaMemcpy(gpu_pairs.data(), d_t2,
                         sizeof(pos2gpu::T2PairingGpu) * gpu_count,
                         cudaMemcpyDeviceToHost));
    }
    cudaFree(d_t2_temp); cudaFree(d_t2_count); cudaFree(d_t2); cudaFree(d_t1_mi); cudaFree(d_t1_meta); cudaFree(d_t1);

    std::vector<T2Key> gpu_keys;
    gpu_keys.reserve(gpu_pairs.size());
    for (auto const& p : gpu_pairs) {
        gpu_keys.push_back({p.match_info, p.x_bits, p.meta});
    }
    std::sort(gpu_keys.begin(), gpu_keys.end());

    std::printf("  GPU T2 produced %zu T2Pairings\n", gpu_keys.size());

    if (cpu_keys.size() != gpu_keys.size()) {
        std::printf("  FAIL: count mismatch (CPU %zu vs GPU %zu)\n",
                    cpu_keys.size(), gpu_keys.size());
        return false;
    }

    uint64_t mismatches = 0;
    for (size_t i = 0; i < cpu_keys.size(); ++i) {
        if (!(cpu_keys[i] == gpu_keys[i])) {
            if (mismatches < 5) {
                std::printf("  MISMATCH at i=%zu  cpu=(mi=0x%08x xb=0x%08x meta=0x%016llx)  "
                            "gpu=(mi=0x%08x xb=0x%08x meta=0x%016llx)\n",
                            i,
                            cpu_keys[i].mi, cpu_keys[i].xb, (unsigned long long)cpu_keys[i].meta,
                            gpu_keys[i].mi, gpu_keys[i].xb, (unsigned long long)gpu_keys[i].meta);
            }
            ++mismatches;
        }
    }

    if (mismatches == 0) {
        std::printf("  OK  %zu / %zu T2Pairings match\n", cpu_keys.size(), cpu_keys.size());
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
    {
        std::array<uint8_t, 32> id{};
        for (auto& b : id) b = 0xab;
        all_ok = run_for_id(id, "plot_id=0xab*32", /*k=*/18, /*strength=*/2) && all_ok;
    }
    for (uint32_t seed : {
            1u, 2u, 3u, 5u, 7u, 11u, 13u, 17u,
            19u, 23u, 29u, 31u, 42u, 1337u,
            0xCAFEBABEu, 0xDEADBEEFu,
            0u, 0xFFFFFFFFu, 0x80000000u,
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
