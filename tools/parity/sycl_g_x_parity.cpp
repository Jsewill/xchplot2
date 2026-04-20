// sycl_g_x_parity — validates the SYCL-compiled AES g_x_smem against the
// same function run on the host. Both compile from the same C++ source in
// AesHashGpu.cuh (the _smem family, now fully portable behind the
// PortableAttrs macros), but one goes through acpp's SSCP backend into a
// device kernel and the other through the host C++ compiler. Any
// codegen-introduced divergence shows up byte-by-byte here.
//
// For x in [0, 1<<k):
//     ref    = g_x_smem on the host with the same AES keys + T-tables
//     actual = g_x_smem inside a SYCL parallel_for, reading the same
//              T-tables from a USM buffer
//
// Pass criterion: ref == actual as a memcmp'd array. Slice-4 of the
// SYCL port — exercises the real AES math on the SYCL device for the
// first time, without the complexity of match_all_buckets around it.

#include "gpu/AesHashGpu.cuh"
#include "gpu/AesTables.inl"

#include <sycl/sycl.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

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

// Build the 4×256 uint32_t sT layout the _smem AES functions expect,
// pulling the values from AesTables.inl so the same data feeds both
// the host reference and the device buffer.
std::vector<uint32_t> build_sT()
{
    std::vector<uint32_t> sT(4 * 256);
    for (int i = 0; i < 256; ++i) {
        sT[0 * 256 + i] = pos2gpu::aes_tables::T0[i];
        sT[1 * 256 + i] = pos2gpu::aes_tables::T1[i];
        sT[2 * 256 + i] = pos2gpu::aes_tables::T2[i];
        sT[3 * 256 + i] = pos2gpu::aes_tables::T3[i];
    }
    return sT;
}

bool run_for(sycl::queue& q, uint32_t seed, int k)
{
    uint64_t const N = 1ull << k;
    auto plot_id = derive_plot_id(seed);
    auto keys    = pos2gpu::make_keys(plot_id.data());
    auto sT_host = build_sT();

    std::vector<uint32_t> ref(N);
    for (uint64_t x = 0; x < N; ++x) {
        ref[x] = pos2gpu::g_x_smem(keys, static_cast<uint32_t>(x), k, sT_host.data());
    }

    uint32_t* d_sT  = sycl::malloc_device<uint32_t>(4 * 256, q);
    uint32_t* d_out = sycl::malloc_device<uint32_t>(N, q);
    q.memcpy(d_sT, sT_host.data(), sizeof(uint32_t) * 4 * 256).wait();

    constexpr size_t threads = 256;
    size_t const groups      = (N + threads - 1) / threads;

    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=, keys_copy = keys](sycl::nd_item<1> it) {
            uint64_t x = it.get_global_id(0);
            if (x >= N) return;
            d_out[x] = pos2gpu::g_x_smem(keys_copy, static_cast<uint32_t>(x), k, d_sT);
        }).wait();

    std::vector<uint32_t> actual(N);
    q.memcpy(actual.data(), d_out, sizeof(uint32_t) * N).wait();
    sycl::free(d_sT, q);
    sycl::free(d_out, q);

    if (std::memcmp(ref.data(), actual.data(), sizeof(uint32_t) * N) == 0) {
        std::printf("PASS  seed=%u k=%d N=%llu\n",
                    seed, k, (unsigned long long)N);
        return true;
    }
    for (uint64_t x = 0; x < N; ++x) {
        if (ref[x] != actual[x]) {
            std::fprintf(stderr,
                "FAIL  seed=%u k=%d  x=%llu  ref=0x%08x  actual=0x%08x\n",
                seed, k, (unsigned long long)x, ref[x], actual[x]);
            break;
        }
    }
    return false;
}

} // namespace

int main()
{
    sycl::queue q{ sycl::gpu_selector_v };
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    bool all_pass = true;
    for (uint32_t seed : { 1u, 7u, 31u }) {
        for (int k : { 14, 16, 18 }) {
            if (!run_for(q, seed, k)) all_pass = false;
        }
    }
    return all_pass ? 0 : 1;
}
