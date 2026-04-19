// aes_tezcan_bench — throughput comparison between the current T-table
// AES (pos2gpu::run_rounds_smem, 4 KB smem: T0|T1|T2|T3) and Tezcan-style
// variants (single T0 + __byte_perm for T1/T2/T3, optionally bank-
// replicated to eliminate smem bank conflicts).
//
// Parity is verified bit-exactly against the current implementation
// before any timing is reported.

#include "gpu/AesGpu.cuh"
#include "gpu/AesHashGpu.cuh"
#include "gpu/AesTezcan.cuh"

#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using pos2gpu::AesState;
using pos2gpu::AesHashKeys;

namespace {

#define CHECK(call) do {                                                 \
    cudaError_t _e = (call);                                             \
    if (_e != cudaSuccess) {                                             \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(_e));                             \
        std::exit(2);                                                    \
    }                                                                    \
} while (0)

// ----- baseline: current run_rounds_smem (4 tables, 4 KB) -----
__global__ void bench_tt_4table(AesState const* __restrict__ in,
                                AesState const* __restrict__ key_pairs,
                                AesState* __restrict__ out,
                                uint64_t n, int rounds)
{
    __shared__ uint32_t sT[4 * 256];
    pos2gpu::load_aes_tables_smem(sT);
    __syncthreads();

    uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    AesHashKeys keys;
    keys.round_key_1 = key_pairs[(tid / 32) * 2 + 0];
    keys.round_key_2 = key_pairs[(tid / 32) * 2 + 1];
    out[tid] = pos2gpu::run_rounds_smem(in[tid], keys, rounds, sT);
}

// ----- Tezcan: single T0 + __byte_perm, BANK_SIZE replicas -----
template<int BANK_SIZE>
__global__ void bench_tezcan(AesState const* __restrict__ in,
                             AesState const* __restrict__ key_pairs,
                             AesState* __restrict__ out,
                             uint64_t n, int rounds)
{
    __shared__ uint32_t sT0[256 * BANK_SIZE];
    pos2gpu::load_aes_t0_smem_rep<BANK_SIZE>(sT0);
    __syncthreads();

    uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    AesHashKeys keys;
    keys.round_key_1 = key_pairs[(tid / 32) * 2 + 0];
    keys.round_key_2 = key_pairs[(tid / 32) * 2 + 1];
    out[tid] = pos2gpu::run_rounds_smem_tezcan<BANK_SIZE>(in[tid], keys, rounds, sT0);
}

// ----- parity check on the T-table relationship itself -----
// Try all byte-permutation maps from T0 to T1/T2/T3 and find the one
// that matches across all 256 entries. Also dumps T0/T1/T2/T3 at i=0
// for visual confirmation.
bool verify_table_rotations()
{
    uint32_t h_t0[256], h_t1[256], h_t2[256], h_t3[256];
    CHECK(cudaMemcpyFromSymbol(h_t0, pos2gpu::kAesT0, sizeof(h_t0)));
    CHECK(cudaMemcpyFromSymbol(h_t1, pos2gpu::kAesT1, sizeof(h_t1)));
    CHECK(cudaMemcpyFromSymbol(h_t2, pos2gpu::kAesT2, sizeof(h_t2)));
    CHECK(cudaMemcpyFromSymbol(h_t3, pos2gpu::kAesT3, sizeof(h_t3)));

    std::printf("T-table sample at i=0 (hex):\n");
    std::printf("  T0[0]=%08x  T1[0]=%08x  T2[0]=%08x  T3[0]=%08x\n",
                h_t0[0], h_t1[0], h_t2[0], h_t3[0]);
    std::printf("  T0[1]=%08x  T1[1]=%08x  T2[1]=%08x  T3[1]=%08x\n",
                h_t0[1], h_t1[1], h_t2[1], h_t3[1]);

    auto ror = [](uint32_t x, int n) -> uint32_t {
        n &= 31; return (x >> n) | (x << (32 - n));
    };
    auto rol = [](uint32_t x, int n) -> uint32_t {
        n &= 31; return (x << n) | (x >> (32 - n));
    };

    auto check = [&](uint32_t const* Tdst, uint32_t(*f)(uint32_t,int), int amt) -> int {
        int bad = 0;
        for (int b = 0; b < 256; ++b) if (Tdst[b] != f(h_t0[b], amt)) ++bad;
        return bad;
    };

    struct { char const* name; uint32_t(*f)(uint32_t,int); int amt; } cands[] = {
        {"ror8",  +[](uint32_t x,int n){ n&=31; return (x>>n)|(x<<(32-n)); },  8},
        {"ror16", +[](uint32_t x,int n){ n&=31; return (x>>n)|(x<<(32-n)); }, 16},
        {"ror24", +[](uint32_t x,int n){ n&=31; return (x>>n)|(x<<(32-n)); }, 24},
        {"rol8",  +[](uint32_t x,int n){ n&=31; return (x<<n)|(x>>(32-n)); },  8},
        {"rol16", +[](uint32_t x,int n){ n&=31; return (x<<n)|(x>>(32-n)); }, 16},
        {"rol24", +[](uint32_t x,int n){ n&=31; return (x<<n)|(x>>(32-n)); }, 24},
    };
    char const* best1 = nullptr; char const* best2 = nullptr; char const* best3 = nullptr;
    for (auto& c : cands) {
        if (check(h_t1, c.f, c.amt) == 0 && !best1) best1 = c.name;
        if (check(h_t2, c.f, c.amt) == 0 && !best2) best2 = c.name;
        if (check(h_t3, c.f, c.amt) == 0 && !best3) best3 = c.name;
    }
    std::printf("T1 = %s(T0)  T2 = %s(T0)  T3 = %s(T0)\n",
                best1 ? best1 : "NONE", best2 ? best2 : "NONE", best3 ? best3 : "NONE");
    (void)ror; (void)rol;
    return best1 && best2 && best3;
}

double ghash_per_s(uint64_t n_states, float ms)
{
    return double(n_states) / (ms * 1e-3) / 1e9;
}

struct BenchResult {
    char const* name;
    size_t smem_bytes;
    float ms_min;
    bool parity_ok;
};

template<int BANK_SIZE>
float run_tezcan(AesState const* d_in, AesState const* d_keys, AesState* d_out,
                 uint64_t n_states, int rounds, cudaEvent_t e0, cudaEvent_t e1)
{
    constexpr int kThreads = 256;
    uint64_t blocks = (n_states + kThreads - 1) / kThreads;

    // Warmup
    bench_tezcan<BANK_SIZE><<<blocks, kThreads>>>(d_in, d_keys, d_out, n_states, rounds);
    CHECK(cudaDeviceSynchronize());

    float ms_min = 1e30f;
    for (int run = 0; run < 3; ++run) {
        CHECK(cudaEventRecord(e0));
        bench_tezcan<BANK_SIZE><<<blocks, kThreads>>>(d_in, d_keys, d_out, n_states, rounds);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < ms_min) ms_min = ms;
    }
    return ms_min;
}

} // namespace

int main(int argc, char** argv)
{
    pos2gpu::initialize_aes_tables();
    CHECK(cudaDeviceSynchronize());

    if (!verify_table_rotations()) {
        std::fprintf(stderr, "FATAL: T1/T2/T3 are not byte rotations of T0 on this build.\n"
                             "The Tezcan single-table trick does not apply.\n");
        return 3;
    }

    uint64_t n_states = 4ull << 20;
    if (argc > 1) n_states = std::strtoull(argv[1], nullptr, 10);
    n_states = (n_states + 31) & ~uint64_t(31);
    uint64_t n_batches = n_states / 32;
    int rounds = pos2gpu::kAesPairingRounds;

    std::printf("aes_tezcan_bench  n_states=%llu  rounds/hash=%d (=%d AES rounds/hash)\n",
                (unsigned long long)n_states, rounds, rounds * 2);

    std::mt19937_64 rng(0x50C0A53C0C0A);
    std::vector<AesState> h_in(n_states), h_keys(n_batches * 2);
    auto rand32 = [&]() { return uint32_t(rng()); };
    for (auto& s : h_in)   { s.w[0]=rand32(); s.w[1]=rand32(); s.w[2]=rand32(); s.w[3]=rand32(); }
    for (auto& s : h_keys) { s.w[0]=rand32(); s.w[1]=rand32(); s.w[2]=rand32(); s.w[3]=rand32(); }

    AesState *d_in = nullptr, *d_keys = nullptr, *d_baseline = nullptr, *d_test = nullptr;
    CHECK(cudaMalloc(&d_in,       sizeof(AesState) * n_states));
    CHECK(cudaMalloc(&d_keys,     sizeof(AesState) * n_batches * 2));
    CHECK(cudaMalloc(&d_baseline, sizeof(AesState) * n_states));
    CHECK(cudaMalloc(&d_test,     sizeof(AesState) * n_states));
    CHECK(cudaMemcpy(d_in,   h_in.data(),   sizeof(AesState) * n_states,      cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_keys, h_keys.data(), sizeof(AesState) * n_batches * 2, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));

    // ---- baseline ----
    constexpr int kThreads = 256;
    uint64_t blocks = (n_states + kThreads - 1) / kThreads;
    bench_tt_4table<<<blocks, kThreads>>>(d_in, d_keys, d_baseline, n_states, rounds);
    CHECK(cudaDeviceSynchronize());
    float base_ms = 1e30f;
    for (int run = 0; run < 3; ++run) {
        CHECK(cudaEventRecord(e0));
        bench_tt_4table<<<blocks, kThreads>>>(d_in, d_keys, d_baseline, n_states, rounds);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < base_ms) base_ms = ms;
    }

    std::vector<AesState> h_baseline(n_states);
    CHECK(cudaMemcpy(h_baseline.data(), d_baseline, sizeof(AesState) * n_states, cudaMemcpyDeviceToHost));

    // ---- Tezcan variants ----
    auto run_and_check = [&](char const* name, size_t smem, float ms) {
        std::vector<AesState> h_test(n_states);
        CHECK(cudaMemcpy(h_test.data(), d_test, sizeof(AesState) * n_states, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (uint64_t i = 0; i < n_states && bad < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (h_test[i].w[j] != h_baseline[i].w[j]) { ++bad; break; }
            }
        }
        return BenchResult{name, smem, ms, bad == 0};
    };

    std::vector<BenchResult> results;
    results.push_back(BenchResult{"T-table 4x (current run_rounds_smem)", 4*256*sizeof(uint32_t), base_ms, true});

    float m1  = run_tezcan<1 >(d_in, d_keys, d_test, n_states, rounds, e0, e1);
    results.push_back(run_and_check("Tezcan 1x (single T0 + byte_perm)",   1*256*sizeof(uint32_t), m1));

    float m4  = run_tezcan<4 >(d_in, d_keys, d_test, n_states, rounds, e0, e1);
    results.push_back(run_and_check("Tezcan 4x (4-replica + byte_perm)",   4*256*sizeof(uint32_t), m4));

    float m8  = run_tezcan<8 >(d_in, d_keys, d_test, n_states, rounds, e0, e1);
    results.push_back(run_and_check("Tezcan 8x (8-replica + byte_perm)",   8*256*sizeof(uint32_t), m8));

    float m16 = run_tezcan<16>(d_in, d_keys, d_test, n_states, rounds, e0, e1);
    results.push_back(run_and_check("Tezcan 16x (16-replica + byte_perm)", 16*256*sizeof(uint32_t), m16));

    float m32 = run_tezcan<32>(d_in, d_keys, d_test, n_states, rounds, e0, e1);
    results.push_back(run_and_check("Tezcan 32x (32-replica + byte_perm)", 32*256*sizeof(uint32_t), m32));

    std::printf("\n%-45s  %7s  %8s  %10s  %9s  %s\n",
                "variant", "smem", "ms", "Ghash/s", "vs base", "parity");
    std::printf("%-45s  %7s  %8s  %10s  %9s  %s\n",
                "-------", "----", "--", "-------", "-------", "------");
    for (auto const& r : results) {
        double rate = ghash_per_s(n_states, r.ms_min);
        double speedup = double(base_ms) / double(r.ms_min);
        std::printf("%-45s  %5zuB  %8.3f  %10.3f  %8.3fx  %s\n",
                    r.name, r.smem_bytes, r.ms_min, rate, speedup,
                    r.parity_ok ? "OK" : "FAIL");
    }

    cudaFree(d_in); cudaFree(d_keys); cudaFree(d_baseline); cudaFree(d_test);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}
