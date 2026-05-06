// sycl_t3_phase_sharded_parity — exercises MultiGpuPlotPipeline's T3
// phase (Phase 2.3c) with N=2 virtual shards on the same physical queue
// and verifies the union of per-shard sorted u64 proof_fragments is
// byte-identical (as a multiset) to the single-GPU T3 phase output.
//
// T3 emits T3PairingGpu = single u64 proof_fragment via atomic cursor;
// post-sort tie order isn't deterministic between the reference and
// sharded paths, so the comparison sorts both arrays before memcmp.

#include "gpu/AesHashGpu.cuh"
#include "gpu/PipelineKernels.cuh"
#include "gpu/Sort.cuh"
#include "gpu/SyclBackend.hpp"
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T3Kernel.cuh"
#include "gpu/XsCandidateGpu.hpp"
#include "gpu/XsKernels.cuh"
#include "host/BatchPlotter.hpp"
#include "host/MultiGpuPlotPipeline.hpp"
#include "host/PoolSizing.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// Reference: full single-GPU pipeline through Xs + T1 + T1-sort + T2 +
// T2-sort + T3 + T3-sort, returning the sorted proof_fragment stream.
std::vector<std::uint64_t> single_gpu_t3_phase(pos2gpu::BatchEntry const& entry)
{
    auto& q = pos2gpu::sycl_backend::queue();
    int const k = entry.k;
    std::uint64_t const total_xs = std::uint64_t{1} << k;
    std::uint32_t const xor_const = entry.testnet ? 0xA3B1C4D7u : 0u;
    pos2gpu::AesHashKeys const keys =
        pos2gpu::make_keys(entry.plot_id.data());

    // ----- Xs phase. -----
    std::uint32_t* d_keys_a = sycl::malloc_device<std::uint32_t>(total_xs, q);
    std::uint32_t* d_vals_a = sycl::malloc_device<std::uint32_t>(total_xs, q);
    std::uint32_t* d_keys_b = sycl::malloc_device<std::uint32_t>(total_xs, q);
    std::uint32_t* d_vals_b = sycl::malloc_device<std::uint32_t>(total_xs, q);
    auto* d_xs = sycl::malloc_device<pos2gpu::XsCandidateGpu>(total_xs, q);

    pos2gpu::launch_xs_gen(keys, d_keys_a, d_vals_a, total_xs, k, xor_const, q);

    std::size_t xs_sort_bytes = 0;
    pos2gpu::launch_sort_pairs_u32_u32(
        nullptr, xs_sort_bytes, nullptr, nullptr, nullptr, nullptr,
        total_xs, 0, k, q);
    void* d_xs_scratch = xs_sort_bytes
        ? sycl::malloc_device(xs_sort_bytes, q) : nullptr;
    pos2gpu::launch_sort_pairs_u32_u32(
        d_xs_scratch ? d_xs_scratch
                     : reinterpret_cast<void*>(std::uintptr_t{1}),
        xs_sort_bytes,
        d_keys_a, d_keys_b, d_vals_a, d_vals_b,
        total_xs, 0, k, q);
    pos2gpu::launch_xs_pack(d_keys_b, d_vals_b, d_xs, total_xs, q);
    q.wait();
    if (d_xs_scratch) sycl::free(d_xs_scratch, q);
    sycl::free(d_keys_a, q); sycl::free(d_vals_a, q);
    sycl::free(d_keys_b, q); sycl::free(d_vals_b, q);

    // ----- T1 match + sort. -----
    auto t1p = pos2gpu::make_t1_params(k, entry.strength);
    std::uint64_t const t1_cap =
        pos2gpu::match_phase_capacity(k, t1p.num_section_bits);
    auto* d_t1_meta  = sycl::malloc_device<std::uint64_t>(t1_cap, q);
    auto* d_t1_mi    = sycl::malloc_device<std::uint32_t>(t1_cap, q);
    auto* d_t1_count = sycl::malloc_device<std::uint64_t>(1, q);

    std::size_t t1_temp_bytes = 0;
    pos2gpu::launch_t1_match(
        entry.plot_id.data(), t1p, d_xs, total_xs,
        nullptr, nullptr, nullptr, t1_cap,
        nullptr, &t1_temp_bytes, q);
    void* d_t1_temp = t1_temp_bytes
        ? sycl::malloc_device(t1_temp_bytes, q) : nullptr;
    pos2gpu::launch_t1_match(
        entry.plot_id.data(), t1p, d_xs, total_xs,
        d_t1_meta, d_t1_mi, d_t1_count, t1_cap,
        d_t1_temp ? d_t1_temp : reinterpret_cast<void*>(std::uintptr_t{1}),
        &t1_temp_bytes, q);
    q.wait();
    sycl::free(d_xs, q);
    if (d_t1_temp) sycl::free(d_t1_temp, q);

    std::uint64_t t1_count = 0;
    q.memcpy(&t1_count, d_t1_count, sizeof(std::uint64_t)).wait();
    sycl::free(d_t1_count, q);

    auto* d_t1_idx_in   = sycl::malloc_device<std::uint32_t>(t1_count, q);
    auto* d_t1_idx_out  = sycl::malloc_device<std::uint32_t>(t1_count, q);
    auto* d_t1_mi_sorted   = sycl::malloc_device<std::uint32_t>(t1_count, q);
    auto* d_t1_meta_sorted = sycl::malloc_device<std::uint64_t>(t1_count, q);

    std::size_t t1_sort_bytes = 0;
    pos2gpu::launch_sort_pairs_u32_u32(
        nullptr, t1_sort_bytes, nullptr, nullptr, nullptr, nullptr,
        t1_count, 0, k, q);
    void* d_t1_sort_scratch = t1_sort_bytes
        ? sycl::malloc_device(t1_sort_bytes, q) : nullptr;
    pos2gpu::launch_init_u32_identity(d_t1_idx_in, t1_count, q);
    pos2gpu::launch_sort_pairs_u32_u32(
        d_t1_sort_scratch ? d_t1_sort_scratch
                          : reinterpret_cast<void*>(std::uintptr_t{1}),
        t1_sort_bytes,
        d_t1_mi, d_t1_mi_sorted, d_t1_idx_in, d_t1_idx_out,
        t1_count, 0, k, q);
    pos2gpu::launch_gather_u64(
        d_t1_meta, d_t1_idx_out, d_t1_meta_sorted, t1_count, q);
    q.wait();
    if (d_t1_sort_scratch) sycl::free(d_t1_sort_scratch, q);
    sycl::free(d_t1_idx_in,  q);
    sycl::free(d_t1_idx_out, q);
    sycl::free(d_t1_mi,      q);
    sycl::free(d_t1_meta,    q);

    // ----- T2 match + sort. -----
    auto t2p = pos2gpu::make_t2_params(k, entry.strength);
    std::uint64_t const t2_cap =
        pos2gpu::match_phase_capacity(k, t2p.num_section_bits);
    auto* d_t2_meta  = sycl::malloc_device<std::uint64_t>(t2_cap, q);
    auto* d_t2_mi    = sycl::malloc_device<std::uint32_t>(t2_cap, q);
    auto* d_t2_xbits = sycl::malloc_device<std::uint32_t>(t2_cap, q);
    auto* d_t2_count = sycl::malloc_device<std::uint64_t>(1, q);

    std::size_t t2_temp_bytes = 0;
    pos2gpu::launch_t2_match(
        entry.plot_id.data(), t2p, nullptr, nullptr, t1_count,
        nullptr, nullptr, nullptr, d_t2_count, t2_cap,
        nullptr, &t2_temp_bytes, q);
    void* d_t2_temp = t2_temp_bytes
        ? sycl::malloc_device(t2_temp_bytes, q) : nullptr;
    pos2gpu::launch_t2_match(
        entry.plot_id.data(), t2p,
        d_t1_meta_sorted, d_t1_mi_sorted, t1_count,
        d_t2_meta, d_t2_mi, d_t2_xbits, d_t2_count, t2_cap,
        d_t2_temp ? d_t2_temp : reinterpret_cast<void*>(std::uintptr_t{1}),
        &t2_temp_bytes, q);
    q.wait();
    sycl::free(d_t1_mi_sorted,   q);
    sycl::free(d_t1_meta_sorted, q);
    if (d_t2_temp) sycl::free(d_t2_temp, q);

    std::uint64_t t2_count = 0;
    q.memcpy(&t2_count, d_t2_count, sizeof(std::uint64_t)).wait();
    sycl::free(d_t2_count, q);

    auto* d_t2_idx_in   = sycl::malloc_device<std::uint32_t>(t2_count, q);
    auto* d_t2_idx_out  = sycl::malloc_device<std::uint32_t>(t2_count, q);
    auto* d_t2_mi_sorted    = sycl::malloc_device<std::uint32_t>(t2_count, q);
    auto* d_t2_meta_sorted  = sycl::malloc_device<std::uint64_t>(t2_count, q);
    auto* d_t2_xbits_sorted = sycl::malloc_device<std::uint32_t>(t2_count, q);

    std::size_t t2_sort_bytes = 0;
    pos2gpu::launch_sort_pairs_u32_u32(
        nullptr, t2_sort_bytes, nullptr, nullptr, nullptr, nullptr,
        t2_count, 0, k, q);
    void* d_t2_sort_scratch = t2_sort_bytes
        ? sycl::malloc_device(t2_sort_bytes, q) : nullptr;
    pos2gpu::launch_init_u32_identity(d_t2_idx_in, t2_count, q);
    pos2gpu::launch_sort_pairs_u32_u32(
        d_t2_sort_scratch ? d_t2_sort_scratch
                          : reinterpret_cast<void*>(std::uintptr_t{1}),
        t2_sort_bytes,
        d_t2_mi, d_t2_mi_sorted, d_t2_idx_in, d_t2_idx_out,
        t2_count, 0, k, q);
    pos2gpu::launch_permute_t2(
        d_t2_meta, d_t2_xbits, d_t2_idx_out,
        d_t2_meta_sorted, d_t2_xbits_sorted, t2_count, q);
    q.wait();
    if (d_t2_sort_scratch) sycl::free(d_t2_sort_scratch, q);
    sycl::free(d_t2_idx_in,  q);
    sycl::free(d_t2_idx_out, q);
    sycl::free(d_t2_mi,      q);
    sycl::free(d_t2_meta,    q);
    sycl::free(d_t2_xbits,   q);

    // ----- T3 match. -----
    auto t3p = pos2gpu::make_t3_params(k, entry.strength);
    std::uint64_t const t3_cap =
        pos2gpu::match_phase_capacity(k, t3p.num_section_bits);
    auto* d_t3       = sycl::malloc_device<pos2gpu::T3PairingGpu>(t3_cap, q);
    auto* d_t3_count = sycl::malloc_device<std::uint64_t>(1, q);

    std::size_t t3_temp_bytes = 0;
    pos2gpu::launch_t3_match(
        entry.plot_id.data(), t3p,
        nullptr, nullptr, nullptr, t2_count,
        nullptr, d_t3_count, t3_cap,
        nullptr, &t3_temp_bytes, q);
    void* d_t3_temp = t3_temp_bytes
        ? sycl::malloc_device(t3_temp_bytes, q) : nullptr;
    pos2gpu::launch_t3_match(
        entry.plot_id.data(), t3p,
        d_t2_meta_sorted, d_t2_xbits_sorted, d_t2_mi_sorted, t2_count,
        d_t3, d_t3_count, t3_cap,
        d_t3_temp ? d_t3_temp : reinterpret_cast<void*>(std::uintptr_t{1}),
        &t3_temp_bytes, q);
    q.wait();
    sycl::free(d_t2_mi_sorted,    q);
    sycl::free(d_t2_meta_sorted,  q);
    sycl::free(d_t2_xbits_sorted, q);
    if (d_t3_temp) sycl::free(d_t3_temp, q);

    std::uint64_t t3_count = 0;
    q.memcpy(&t3_count, d_t3_count, sizeof(std::uint64_t)).wait();
    sycl::free(d_t3_count, q);

    // ----- T3 sort by proof_fragment (low 2k bits). -----
    auto* d_frags_in  = reinterpret_cast<std::uint64_t*>(d_t3);
    auto* d_frags_out = sycl::malloc_device<std::uint64_t>(t3_count, q);

    std::size_t t3_sort_bytes = 0;
    pos2gpu::launch_sort_keys_u64(
        nullptr, t3_sort_bytes, nullptr, nullptr,
        t3_count, 0, 2 * k, q);
    void* d_t3_sort_scratch = t3_sort_bytes
        ? sycl::malloc_device(t3_sort_bytes, q) : nullptr;
    pos2gpu::launch_sort_keys_u64(
        d_t3_sort_scratch ? d_t3_sort_scratch
                          : reinterpret_cast<void*>(std::uintptr_t{1}),
        t3_sort_bytes,
        d_frags_in, d_frags_out, t3_count, 0, 2 * k, q);
    q.wait();
    if (d_t3_sort_scratch) sycl::free(d_t3_sort_scratch, q);

    std::vector<std::uint64_t> out(t3_count);
    if (t3_count > 0) {
        q.memcpy(out.data(), d_frags_out,
                 t3_count * sizeof(std::uint64_t)).wait();
    }
    sycl::free(d_t3,        q);
    sycl::free(d_frags_out, q);
    return out;
}

std::vector<std::uint64_t> sharded_t3_phase(pos2gpu::BatchEntry const& entry)
{
    auto& q = pos2gpu::sycl_backend::queue();
    pos2gpu::BatchOptions opts{};
    opts.shard_plot = true;

    std::vector<pos2gpu::MultiGpuShardContext> shards(2);
    shards[0] = {&q, 0};
    shards[1] = {&q, 0};

    pos2gpu::MultiGpuPlotPipeline pipeline(entry, opts, std::move(shards));
    pipeline.run_through(pos2gpu::Phase::T3);

    std::uint64_t total = 0;
    for (std::size_t s = 0; s < pipeline.shard_count(); ++s) {
        total += pipeline.t3_phase_count(s);
    }

    std::vector<std::uint64_t> out(total);
    std::uint64_t off = 0;
    for (std::size_t s = 0; s < pipeline.shard_count(); ++s) {
        std::uint64_t const c = pipeline.t3_phase_count(s);
        if (c == 0) continue;
        sycl::queue& sq = pipeline.shard_queue(s);
        sq.memcpy(out.data() + off, pipeline.t3_phase_d_frags(s),
                  c * sizeof(std::uint64_t)).wait();
        off += c;
    }
    return out;
}

bool run_one(int k, bool testnet, std::uint8_t plot_id_seed)
{
    pos2gpu::BatchEntry entry{};
    entry.k          = k;
    entry.strength   = 2;
    entry.plot_index = 0;
    entry.meta_group = 0;
    entry.testnet    = testnet;
    for (int i = 0; i < 32; ++i) {
        entry.plot_id[i] =
            static_cast<std::uint8_t>(plot_id_seed * 17u + i * 19u);
    }

    auto ref     = single_gpu_t3_phase(entry);
    auto sharded = sharded_t3_phase  (entry);

    // SET comparison. T3 emits via atomic cursor; the radix sort by
    // low 2k bits leaves the high bits of each fragment in arbitrary
    // order, so two runs of the same input may produce different byte
    // sequences if any two fragments share their low 2k bits. Sort
    // both arrays by the FULL 64-bit fragment value before memcmp.
    std::sort(ref.begin(),     ref.end());
    std::sort(sharded.begin(), sharded.end());

    bool const size_ok  = (ref.size() == sharded.size());
    bool const bytes_ok = size_ok && std::memcmp(
        ref.data(), sharded.data(),
        sizeof(std::uint64_t) * ref.size()) == 0;
    bool const ok = size_ok && bytes_ok;

    std::printf(
        "%s t3-phase k=%d testnet=%d seed=%u  "
        "[count=%llu vs %llu  size=%d bytes=%d]\n",
        ok ? "PASS" : "FAIL", k, testnet ? 1 : 0,
        static_cast<unsigned>(plot_id_seed),
        static_cast<unsigned long long>(ref.size()),
        static_cast<unsigned long long>(sharded.size()),
        size_ok ? 1 : 0, bytes_ok ? 1 : 0);
    return ok;
}

} // namespace

int main()
{
    bool all_ok = true;
    for (int k : {18, 20, 22}) {
        for (bool testnet : {false, true}) {
            for (std::uint8_t seed : {7u, 31u}) {
                all_ok = run_one(k, testnet, seed) && all_ok;
            }
        }
    }
    return all_ok ? 0 : 1;
}
