// sycl_xs_phase_sharded_parity — exercises MultiGpuPlotPipeline's Xs
// phase (Phase 2.2) with N=2 virtual shards on the same physical queue
// and verifies the union of per-shard XsCandidateGpu outputs is
// byte-identical to the single-GPU Xs phase output (launch_xs_gen +
// launch_sort_pairs_u32_u32 + launch_xs_pack on the full input).
//
// Validates the host-pinned-bounce distributed sort + the per-shard
// launch_xs_gen_range / launch_xs_pack wiring on hardware accessible
// to a 1-GPU dev box; real multi-physical-GPU runs are downstream.

#include "gpu/AesHashGpu.cuh"
#include "gpu/Sort.cuh"
#include "gpu/SyclBackend.hpp"
#include "gpu/XsCandidateGpu.hpp"
#include "gpu/XsKernels.cuh"
#include "host/BatchPlotter.hpp"
#include "host/MultiGpuPlotPipeline.hpp"

#include <sycl/sycl.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

// Reference implementation: single-GPU Xs phase. Produces packed
// XsCandidateGpu array on the host.
std::vector<pos2gpu::XsCandidateGpu> single_gpu_xs_phase(
    pos2gpu::BatchEntry const& entry)
{
    auto& q = pos2gpu::sycl_backend::queue();
    int const k = entry.k;
    uint64_t const total_xs = uint64_t{1} << k;
    uint32_t const xor_const = entry.testnet ? 0xA3B1C4D7u : 0u;
    pos2gpu::AesHashKeys const keys = pos2gpu::make_keys(entry.plot_id.data());

    uint32_t* d_keys_a = sycl::malloc_device<uint32_t>(total_xs, q);
    uint32_t* d_vals_a = sycl::malloc_device<uint32_t>(total_xs, q);
    uint32_t* d_keys_b = sycl::malloc_device<uint32_t>(total_xs, q);
    uint32_t* d_vals_b = sycl::malloc_device<uint32_t>(total_xs, q);
    pos2gpu::XsCandidateGpu* d_xs =
        sycl::malloc_device<pos2gpu::XsCandidateGpu>(total_xs, q);

    pos2gpu::launch_xs_gen(keys, d_keys_a, d_vals_a, total_xs, k, xor_const, q);

    size_t scratch_bytes = 0;
    pos2gpu::launch_sort_pairs_u32_u32(
        nullptr, scratch_bytes, nullptr, nullptr, nullptr, nullptr,
        total_xs, 0, k, q);
    void* d_scratch = scratch_bytes
        ? sycl::malloc_device(scratch_bytes, q) : nullptr;
    pos2gpu::launch_sort_pairs_u32_u32(
        d_scratch ? d_scratch : reinterpret_cast<void*>(uintptr_t{1}),
        scratch_bytes,
        d_keys_a, d_keys_b, d_vals_a, d_vals_b,
        total_xs, 0, k, q);
    pos2gpu::launch_xs_pack(d_keys_b, d_vals_b, d_xs, total_xs, q);
    q.wait();

    std::vector<pos2gpu::XsCandidateGpu> h_xs(total_xs);
    q.memcpy(h_xs.data(), d_xs,
             sizeof(pos2gpu::XsCandidateGpu) * total_xs).wait();

    if (d_scratch) sycl::free(d_scratch, q);
    sycl::free(d_keys_a, q); sycl::free(d_vals_a, q);
    sycl::free(d_keys_b, q); sycl::free(d_vals_b, q);
    sycl::free(d_xs,     q);
    return h_xs;
}

// Sharded implementation: 2 virtual shards on the same queue.
std::vector<pos2gpu::XsCandidateGpu> sharded_xs_phase(
    pos2gpu::BatchEntry const& entry)
{
    auto& q = pos2gpu::sycl_backend::queue();
    pos2gpu::BatchOptions opts{};
    opts.shard_plot = true;
    std::vector<pos2gpu::MultiGpuShardContext> shards(2);
    shards[0] = {&q, 0};
    shards[1] = {&q, 0};

    pos2gpu::MultiGpuPlotPipeline pipeline(entry, opts, std::move(shards));
    pipeline.run_xs_phase();

    // Concatenate per-shard packed outputs in shard-id order.
    uint64_t const total_xs = uint64_t{1} << entry.k;
    std::vector<pos2gpu::XsCandidateGpu> h_xs;
    h_xs.reserve(total_xs);
    for (std::size_t s = 0; s < pipeline.shard_count(); ++s) {
        std::uint64_t const c = pipeline.xs_phase_count(s);
        if (c == 0) continue;
        std::vector<pos2gpu::XsCandidateGpu> shard_buf(c);
        pipeline.shard_queue(s).memcpy(
            shard_buf.data(), pipeline.xs_phase_d_xs(s),
            sizeof(pos2gpu::XsCandidateGpu) * c).wait();
        h_xs.insert(h_xs.end(), shard_buf.begin(), shard_buf.end());
    }
    return h_xs;
}

bool run_one(int k, bool testnet, uint8_t plot_id_seed)
{
    pos2gpu::BatchEntry entry{};
    entry.k = k;
    entry.strength = 2;
    entry.plot_index = 0;
    entry.meta_group = 0;
    entry.testnet = testnet;
    for (int i = 0; i < 32; ++i) {
        entry.plot_id[i] = static_cast<uint8_t>(plot_id_seed * 17u + i * 19u);
    }

    auto ref      = single_gpu_xs_phase(entry);
    auto sharded  = sharded_xs_phase  (entry);

    bool const size_ok = (ref.size() == sharded.size());
    bool const bytes_ok = size_ok && std::memcmp(
        ref.data(), sharded.data(),
        sizeof(pos2gpu::XsCandidateGpu) * ref.size()) == 0;
    bool const ok = size_ok && bytes_ok;
    std::printf("%s xs-phase k=%d testnet=%d seed=%u  [size=%d bytes=%d]\n",
                ok ? "PASS" : "FAIL", k, testnet ? 1 : 0,
                static_cast<unsigned>(plot_id_seed),
                size_ok ? 1 : 0, bytes_ok ? 1 : 0);
    return ok;
}

} // namespace

int main()
{
    bool all_ok = true;
    for (int k : {18, 20, 22}) {
        for (bool testnet : {false, true}) {
            for (uint8_t seed : {7u, 31u}) {
                all_ok = run_one(k, testnet, seed) && all_ok;
            }
        }
    }
    return all_ok ? 0 : 1;
}
