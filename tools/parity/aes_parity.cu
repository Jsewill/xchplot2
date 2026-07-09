// aes_parity — proves the GPU AES round implementation is byte-identical
// to the CPU reference (`AesHash` from pos2-chip) across all four hash
// functions: g_x, pairing, chain, matching_target.
//
// Coverage:
//   - 4 plot_ids (deterministic, derived from a small mixing function).
//   - For each plot_id:
//       g_x:              all x in [0, 262 144)
//       pairing:          16 384 random (meta_l, meta_r) pairs
//       chain:            16 384 random uint64 inputs
//       matching_target:  16 384 random (table_id, match_key, meta) triples
//   - All round-counts use the constants pos2-chip ships (16 rounds).
//
// Pass criterion: every output bit matches.

#include "gpu/AesGpu.cuh"
#include "gpu/AesHashGpu.cuh"

#include "pos/aes/AesHash.hpp"
#include "pos/aes/intrin_portable.h"

#include "ParityCommon.hpp"

#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using pos2gpu::parity::derive_plot_id;
using pos2gpu::parity::Stats;
using pos2gpu::parity::compare;

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(2); \
    } \
} while (0)

constexpr int kRounds = 16;
constexpr int kK      = 18;
constexpr uint64_t kGxSamples           = 1ULL << 18; // 262 144
constexpr uint64_t kRandomSamples       =       16384;

// ---- g_x ----------------------------------------------------------------

__global__ void g_x_kernel(pos2gpu::AesHashKeys keys, uint32_t* out, uint64_t total, int k)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    out[idx] = pos2gpu::g_x(keys, static_cast<uint32_t>(idx), k, kRounds);
}

// ---- pairing ------------------------------------------------------------

struct PairingIn { uint64_t l, r; };
struct PairingOut { uint32_t r[4]; };

__global__ void pairing_kernel(
    pos2gpu::AesHashKeys keys, PairingIn const* in, PairingOut* out, uint64_t total)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    auto p = pos2gpu::pairing(keys, in[idx].l, in[idx].r, /*extra_rounds_bits=*/0);
    PairingOut o; o.r[0] = p.r[0]; o.r[1] = p.r[1]; o.r[2] = p.r[2]; o.r[3] = p.r[3];
    out[idx] = o;
}

// ---- chain --------------------------------------------------------------

__global__ void chain_kernel(
    pos2gpu::AesHashKeys keys, uint64_t const* in, uint64_t* out, uint64_t total)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    out[idx] = pos2gpu::chain(keys, in[idx]);
}

// ---- matching_target ----------------------------------------------------

struct MtIn { uint32_t table_id; uint32_t match_key; uint64_t meta; };

__global__ void mt_kernel(
    pos2gpu::AesHashKeys keys, MtIn const* in, uint32_t* out, uint64_t total)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    out[idx] = pos2gpu::matching_target(
        keys, in[idx].table_id, in[idx].match_key, in[idx].meta, /*extra_rounds_bits=*/0);
}

// ---- shared-memory wrappers (ttable4 + tezcan16) ------------------------
// Inventory of hot-kernel entry points: g_x_smem (Xs), pairing_smem +
// matching_target_smem (T1/T2/T3). chain_smem is unused by those kernels
// but covered here so a future caller cannot silently diverge.
//
// The *_ttable4_kernel entry points call the DISPATCHING wrappers
// (g_x_smem etc.), so their smem must use the config macros: under
// XCHPLOT2_AES_ROUND=tezcan16 the wrappers take the Tezcan path and a
// hard-coded 4 KB buffer would be read out of bounds. In default builds
// they exercise ttable4 exactly as the name says.

__global__ void g_x_smem_ttable4_kernel(
    pos2gpu::AesHashKeys keys, uint32_t* out, uint64_t total, int k)
{
    XCHPLOT2_AES_SMEM_DECL(sT);
    XCHPLOT2_AES_SMEM_LOAD(sT);
    __syncthreads();
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    out[idx] = pos2gpu::g_x_smem(keys, static_cast<uint32_t>(idx), k, sT, kRounds);
}

template<int BANK>
__global__ void g_x_smem_tezcan_kernel(
    pos2gpu::AesHashKeys keys, uint32_t* out, uint64_t total, int k)
{
    __shared__ uint32_t sT0[256 * BANK];
    pos2gpu::load_aes_t0_smem_rep<BANK>(sT0);
    __syncthreads();
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    out[idx] = pos2gpu::g_x_smem_tezcan<BANK>(
        keys, static_cast<uint32_t>(idx), k, sT0, kRounds);
}

__global__ void pairing_smem_ttable4_kernel(
    pos2gpu::AesHashKeys keys, PairingIn const* in, PairingOut* out, uint64_t total)
{
    XCHPLOT2_AES_SMEM_DECL(sT);
    XCHPLOT2_AES_SMEM_LOAD(sT);
    __syncthreads();
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    auto p = pos2gpu::pairing_smem(keys, in[idx].l, in[idx].r, sT, 0);
    PairingOut o; o.r[0] = p.r[0]; o.r[1] = p.r[1]; o.r[2] = p.r[2]; o.r[3] = p.r[3];
    out[idx] = o;
}

template<int BANK>
__global__ void pairing_smem_tezcan_kernel(
    pos2gpu::AesHashKeys keys, PairingIn const* in, PairingOut* out, uint64_t total)
{
    __shared__ uint32_t sT0[256 * BANK];
    pos2gpu::load_aes_t0_smem_rep<BANK>(sT0);
    __syncthreads();
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    auto p = pos2gpu::pairing_smem_tezcan<BANK>(keys, in[idx].l, in[idx].r, sT0, 0);
    PairingOut o; o.r[0] = p.r[0]; o.r[1] = p.r[1]; o.r[2] = p.r[2]; o.r[3] = p.r[3];
    out[idx] = o;
}

__global__ void mt_smem_ttable4_kernel(
    pos2gpu::AesHashKeys keys, MtIn const* in, uint32_t* out, uint64_t total)
{
    XCHPLOT2_AES_SMEM_DECL(sT);
    XCHPLOT2_AES_SMEM_LOAD(sT);
    __syncthreads();
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    out[idx] = pos2gpu::matching_target_smem(
        keys, in[idx].table_id, in[idx].match_key, in[idx].meta, sT, 0);
}

template<int BANK>
__global__ void mt_smem_tezcan_kernel(
    pos2gpu::AesHashKeys keys, MtIn const* in, uint32_t* out, uint64_t total)
{
    __shared__ uint32_t sT0[256 * BANK];
    pos2gpu::load_aes_t0_smem_rep<BANK>(sT0);
    __syncthreads();
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    out[idx] = pos2gpu::matching_target_smem_tezcan<BANK>(
        keys, in[idx].table_id, in[idx].match_key, in[idx].meta, sT0, 0);
}

__global__ void chain_smem_ttable4_kernel(
    pos2gpu::AesHashKeys keys, uint64_t const* in, uint64_t* out, uint64_t total)
{
    XCHPLOT2_AES_SMEM_DECL(sT);
    XCHPLOT2_AES_SMEM_LOAD(sT);
    __syncthreads();
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    out[idx] = pos2gpu::chain_smem(keys, in[idx], sT);
}

// ---- harness helpers ----------------------------------------------------

template <typename T>
T* device_alloc_and_copy(std::vector<T> const& host)
{
    T* d = nullptr;
    CHECK(cudaMalloc(&d, sizeof(T) * host.size()));
    CHECK(cudaMemcpy(d, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
std::vector<T> launch_and_collect(
    void(*launch)(pos2gpu::AesHashKeys, T const*, T*, uint64_t), // unused; we hardwire below
    pos2gpu::AesHashKeys, std::vector<T> const&)
{ static_assert(sizeof(T) == 0, "use a per-kernel collector"); return {}; }

// One-shot: run a kernel `K` over `n` items, return host vector.
#define COLLECT(KernelLaunch, N, OutT)                                      \
    [&] {                                                                   \
        OutT* d_out = nullptr;                                              \
        CHECK(cudaMalloc(&d_out, sizeof(OutT) * (N)));                      \
        constexpr int kThreads = 256;                                       \
        unsigned blocks = (unsigned)(((N) + kThreads - 1) / kThreads);      \
        KernelLaunch;                                                       \
        CHECK(cudaGetLastError());                                          \
        CHECK(cudaDeviceSynchronize());                                     \
        std::vector<OutT> out(N);                                           \
        CHECK(cudaMemcpy(out.data(), d_out, sizeof(OutT) * (N), cudaMemcpyDeviceToHost)); \
        CHECK(cudaFree(d_out));                                             \
        return out;                                                         \
    }()

// Per-plot-id full sweep.
bool run_for_plot_id(uint32_t seed)
{
    auto id = derive_plot_id(seed);
    auto gpu_keys = pos2gpu::make_keys(id.data());
    AesHash cpu(id.data(), kK);

    std::printf("[plot_id seed=%u  bytes 0..7=", seed);
    for (int i = 0; i < 8; ++i) std::printf("%02x", id[i]);
    std::printf("]\n");

    bool all_ok = true;

    // ---- g_x ----
    {
        std::vector<uint32_t> cpu_out(kGxSamples);
        for (uint64_t x = 0; x < kGxSamples; ++x)
            cpu_out[x] = cpu.g_x<true>(static_cast<uint32_t>(x), kRounds);

        auto gpu_out = COLLECT((g_x_kernel<<<blocks, kThreads>>>(gpu_keys, d_out, kGxSamples, kK)),
                               kGxSamples, uint32_t);
        auto s = compare(kGxSamples, [&](uint64_t i){ return cpu_out[i] == gpu_out[i]; },
                         "g_x", seed);
        std::printf("  g_x:             %llu / %llu  %s\n",
                    static_cast<unsigned long long>(s.total - s.mismatches),
                    static_cast<unsigned long long>(s.total),
                    s.ok() ? "OK" : "FAIL");
        all_ok = all_ok && s.ok();
    }

    // Common deterministic RNG so CPU/GPU consume the same inputs.
    std::mt19937_64 rng(0xC1A5DEADBEEF0000ULL ^ seed);

    // ---- pairing ----
    {
        std::vector<PairingIn>  in(kRandomSamples);
        std::vector<PairingOut> cpu_out(kRandomSamples);
        for (uint64_t i = 0; i < kRandomSamples; ++i) {
            in[i].l = rng();
            in[i].r = rng();
            auto p = cpu.pairing<true>(in[i].l, in[i].r, /*extra_rounds_bits=*/0);
            cpu_out[i].r[0] = p.r[0]; cpu_out[i].r[1] = p.r[1];
            cpu_out[i].r[2] = p.r[2]; cpu_out[i].r[3] = p.r[3];
        }
        auto* d_in = device_alloc_and_copy(in);
        auto gpu_out = COLLECT((pairing_kernel<<<blocks, kThreads>>>(gpu_keys, d_in, d_out, kRandomSamples)),
                               kRandomSamples, PairingOut);
        CHECK(cudaFree(d_in));
        auto s = compare(kRandomSamples, [&](uint64_t i) {
            for (int j = 0; j < 4; ++j) if (cpu_out[i].r[j] != gpu_out[i].r[j]) return false;
            return true;
        }, "pairing", seed);
        std::printf("  pairing:         %llu / %llu  %s\n",
                    static_cast<unsigned long long>(s.total - s.mismatches),
                    static_cast<unsigned long long>(s.total),
                    s.ok() ? "OK" : "FAIL");
        all_ok = all_ok && s.ok();
    }

    // ---- chain ----
    {
        std::vector<uint64_t> in(kRandomSamples);
        std::vector<uint64_t> cpu_out(kRandomSamples);
        for (uint64_t i = 0; i < kRandomSamples; ++i) {
            in[i] = rng();
            cpu_out[i] = cpu.chain<true>(in[i]);
        }
        auto* d_in = device_alloc_and_copy(in);
        auto gpu_out = COLLECT((chain_kernel<<<blocks, kThreads>>>(gpu_keys, d_in, d_out, kRandomSamples)),
                               kRandomSamples, uint64_t);
        CHECK(cudaFree(d_in));
        auto s = compare(kRandomSamples, [&](uint64_t i){ return cpu_out[i] == gpu_out[i]; },
                         "chain", seed);
        std::printf("  chain:           %llu / %llu  %s\n",
                    static_cast<unsigned long long>(s.total - s.mismatches),
                    static_cast<unsigned long long>(s.total),
                    s.ok() ? "OK" : "FAIL");
        all_ok = all_ok && s.ok();
    }

    // ---- matching_target ----
    {
        std::vector<MtIn>     in(kRandomSamples);
        std::vector<uint32_t> cpu_out(kRandomSamples);
        for (uint64_t i = 0; i < kRandomSamples; ++i) {
            in[i].table_id  = uint32_t(rng() % 3) + 1; // 1..3
            in[i].match_key = uint32_t(rng());
            in[i].meta      = rng();
            cpu_out[i] = cpu.matching_target<true>(
                in[i].table_id, in[i].match_key, in[i].meta, /*extra_rounds_bits=*/0);
        }
        auto* d_in = device_alloc_and_copy(in);
        auto gpu_out = COLLECT((mt_kernel<<<blocks, kThreads>>>(gpu_keys, d_in, d_out, kRandomSamples)),
                               kRandomSamples, uint32_t);
        CHECK(cudaFree(d_in));
        auto s = compare(kRandomSamples, [&](uint64_t i){ return cpu_out[i] == gpu_out[i]; },
                         "matching_target", seed);
        std::printf("  matching_target: %llu / %llu  %s\n",
                    static_cast<unsigned long long>(s.total - s.mismatches),
                    static_cast<unsigned long long>(s.total),
                    s.ok() ? "OK" : "FAIL");
        all_ok = all_ok && s.ok();
    }

    // ---- smem wrappers vs soft-AES (ttable4 + tezcan16) ----
    // Reuse the same deterministic inputs as the constant-memory paths
    // above by reseeding the RNG to the same stream start.
    {
        std::mt19937_64 rng_smem(0xC1A5DEADBEEF0000ULL ^ seed);

        // g_x_smem ttable4 + tezcan16
        {
            std::vector<uint32_t> cpu_out(kGxSamples);
            for (uint64_t x = 0; x < kGxSamples; ++x)
                cpu_out[x] = cpu.g_x<true>(static_cast<uint32_t>(x), kRounds);

            auto gpu_tt = COLLECT(
                (g_x_smem_ttable4_kernel<<<blocks, kThreads>>>(gpu_keys, d_out, kGxSamples, kK)),
                kGxSamples, uint32_t);
            auto s_tt = compare(kGxSamples, [&](uint64_t i){ return cpu_out[i] == gpu_tt[i]; },
                                "g_x_smem_ttable4", seed);
            std::printf("  g_x_smem_ttable4: %llu / %llu  %s\n",
                        static_cast<unsigned long long>(s_tt.total - s_tt.mismatches),
                        static_cast<unsigned long long>(s_tt.total),
                        s_tt.ok() ? "OK" : "FAIL");
            all_ok = all_ok && s_tt.ok();

            auto gpu_tz = COLLECT(
                (g_x_smem_tezcan_kernel<16><<<blocks, kThreads>>>(gpu_keys, d_out, kGxSamples, kK)),
                kGxSamples, uint32_t);
            auto s_tz = compare(kGxSamples, [&](uint64_t i){ return cpu_out[i] == gpu_tz[i]; },
                                "g_x_smem_tezcan16", seed);
            std::printf("  g_x_smem_tezcan16:%llu / %llu  %s\n",
                        static_cast<unsigned long long>(s_tz.total - s_tz.mismatches),
                        static_cast<unsigned long long>(s_tz.total),
                        s_tz.ok() ? "OK" : "FAIL");
            all_ok = all_ok && s_tz.ok();
        }

        // pairing_smem
        {
            std::vector<PairingIn>  in(kRandomSamples);
            std::vector<PairingOut> cpu_out(kRandomSamples);
            for (uint64_t i = 0; i < kRandomSamples; ++i) {
                in[i].l = rng_smem();
                in[i].r = rng_smem();
                auto p = cpu.pairing<true>(in[i].l, in[i].r, /*extra_rounds_bits=*/0);
                cpu_out[i].r[0] = p.r[0]; cpu_out[i].r[1] = p.r[1];
                cpu_out[i].r[2] = p.r[2]; cpu_out[i].r[3] = p.r[3];
            }
            auto* d_in = device_alloc_and_copy(in);
            auto gpu_tt = COLLECT(
                (pairing_smem_ttable4_kernel<<<blocks, kThreads>>>(gpu_keys, d_in, d_out, kRandomSamples)),
                kRandomSamples, PairingOut);
            auto s_tt = compare(kRandomSamples, [&](uint64_t i) {
                for (int j = 0; j < 4; ++j) if (cpu_out[i].r[j] != gpu_tt[i].r[j]) return false;
                return true;
            }, "pairing_smem_ttable4", seed);
            std::printf("  pairing_smem_tt: %llu / %llu  %s\n",
                        static_cast<unsigned long long>(s_tt.total - s_tt.mismatches),
                        static_cast<unsigned long long>(s_tt.total),
                        s_tt.ok() ? "OK" : "FAIL");
            all_ok = all_ok && s_tt.ok();

            auto gpu_tz = COLLECT(
                (pairing_smem_tezcan_kernel<16><<<blocks, kThreads>>>(gpu_keys, d_in, d_out, kRandomSamples)),
                kRandomSamples, PairingOut);
            CHECK(cudaFree(d_in));
            auto s_tz = compare(kRandomSamples, [&](uint64_t i) {
                for (int j = 0; j < 4; ++j) if (cpu_out[i].r[j] != gpu_tz[i].r[j]) return false;
                return true;
            }, "pairing_smem_tezcan16", seed);
            std::printf("  pairing_smem_tz: %llu / %llu  %s\n",
                        static_cast<unsigned long long>(s_tz.total - s_tz.mismatches),
                        static_cast<unsigned long long>(s_tz.total),
                        s_tz.ok() ? "OK" : "FAIL");
            all_ok = all_ok && s_tz.ok();
        }

        // chain_smem (ttable4 only — unused by hot kernels, still covered)
        {
            std::vector<uint64_t> in(kRandomSamples);
            std::vector<uint64_t> cpu_out(kRandomSamples);
            for (uint64_t i = 0; i < kRandomSamples; ++i) {
                in[i] = rng_smem();
                cpu_out[i] = cpu.chain<true>(in[i]);
            }
            auto* d_in = device_alloc_and_copy(in);
            auto gpu_out = COLLECT(
                (chain_smem_ttable4_kernel<<<blocks, kThreads>>>(gpu_keys, d_in, d_out, kRandomSamples)),
                kRandomSamples, uint64_t);
            CHECK(cudaFree(d_in));
            auto s = compare(kRandomSamples, [&](uint64_t i){ return cpu_out[i] == gpu_out[i]; },
                             "chain_smem_ttable4", seed);
            std::printf("  chain_smem_tt:   %llu / %llu  %s\n",
                        static_cast<unsigned long long>(s.total - s.mismatches),
                        static_cast<unsigned long long>(s.total),
                        s.ok() ? "OK" : "FAIL");
            all_ok = all_ok && s.ok();
        }

        // matching_target_smem
        {
            std::vector<MtIn>     in(kRandomSamples);
            std::vector<uint32_t> cpu_out(kRandomSamples);
            for (uint64_t i = 0; i < kRandomSamples; ++i) {
                in[i].table_id  = uint32_t(rng_smem() % 3) + 1;
                in[i].match_key = uint32_t(rng_smem());
                in[i].meta      = rng_smem();
                cpu_out[i] = cpu.matching_target<true>(
                    in[i].table_id, in[i].match_key, in[i].meta, /*extra_rounds_bits=*/0);
            }
            auto* d_in = device_alloc_and_copy(in);
            auto gpu_tt = COLLECT(
                (mt_smem_ttable4_kernel<<<blocks, kThreads>>>(gpu_keys, d_in, d_out, kRandomSamples)),
                kRandomSamples, uint32_t);
            auto s_tt = compare(kRandomSamples, [&](uint64_t i){ return cpu_out[i] == gpu_tt[i]; },
                                "mt_smem_ttable4", seed);
            std::printf("  mt_smem_ttable4: %llu / %llu  %s\n",
                        static_cast<unsigned long long>(s_tt.total - s_tt.mismatches),
                        static_cast<unsigned long long>(s_tt.total),
                        s_tt.ok() ? "OK" : "FAIL");
            all_ok = all_ok && s_tt.ok();

            auto gpu_tz = COLLECT(
                (mt_smem_tezcan_kernel<16><<<blocks, kThreads>>>(gpu_keys, d_in, d_out, kRandomSamples)),
                kRandomSamples, uint32_t);
            CHECK(cudaFree(d_in));
            auto s_tz = compare(kRandomSamples, [&](uint64_t i){ return cpu_out[i] == gpu_tz[i]; },
                                "mt_smem_tezcan16", seed);
            std::printf("  mt_smem_tezcan16:%llu / %llu  %s\n",
                        static_cast<unsigned long long>(s_tz.total - s_tz.mismatches),
                        static_cast<unsigned long long>(s_tz.total),
                        s_tz.ok() ? "OK" : "FAIL");
            all_ok = all_ok && s_tz.ok();
        }
    }

    return all_ok;
}

} // namespace

int main()
{
    pos2gpu::initialize_aes_tables();

    bool all_ok = true;
    for (uint32_t seed : {1u, 2u, 17u, 0xCAFE'BABEu}) {
        all_ok = run_for_plot_id(seed) && all_ok;
        std::printf("\n");
    }

    if (all_ok) {
        std::printf("==> ALL OK\n");
        return 0;
    } else {
        std::printf("==> FAIL\n");
        return 1;
    }
}
