// sycl_t2_phase_sharded_parity — exercises MultiGpuPlotPipeline's T2
// phase (Phase 2.3b) with N=2 virtual shards on the same physical queue
// and verifies the union of per-shard sorted (mi, meta, xbits) outputs
// is byte-identical (as a multiset) to the single-GPU T2 phase output.
//
// Validates the full Phase 2.3b stack: replicate T1 mi+meta streams,
// per-shard launch_t2_match_prepare + launch_t2_match_range over the
// shard's bucket subset, and the new launch_sort_pairs_u32_u64u32_dist-
// ributed bouncing T2 (mi, meta, xbits) across the host-pinned bridge.
//
// As with the T1 parity test, T2 emits via an atomic cursor so same-mi
// tie order isn't deterministic; comparison sorts both arrays by the
// full (mi, meta, xbits) tuple before memcmp.

#include "gpu/AesHashGpu.cuh"
#include "gpu/PipelineKernels.cuh"
#include "gpu/Sort.cuh"
#include "gpu/SyclBackend.hpp"
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"
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

struct T2Sorted {
    std::vector<std::uint32_t> mi;
    std::vector<std::uint64_t> meta;
    std::vector<std::uint32_t> xbits;
};

// Reference: full single-GPU Xs + T1 + T1-sort + T2 + T2-sort using the
// same kernel primitives the production GpuPipeline.cpp uses.
T2Sorted single_gpu_t2_phase(pos2gpu::BatchEntry const& entry)
{
    auto& q = pos2gpu::sycl_backend::queue();
    int const k = entry.k;
    std::uint64_t const total_xs = std::uint64_t{1} << k;
    std::uint32_t const xor_const = entry.testnet ? 0xA3B1C4D7u : 0u;
    pos2gpu::AesHashKeys const keys =
        pos2gpu::make_keys(entry.plot_id.data());

    // ----- Xs phase: gen + sort + pack. -----
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

    // ----- T1 match. -----
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

    // ----- T1 sort by mi. -----
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

    // ----- T2 match. -----
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

    // ----- T2 sort by mi (identity sort + permute_t2 fused gather). -----
    auto* d_t2_idx_in   = sycl::malloc_device<std::uint32_t>(t2_count, q);
    auto* d_t2_idx_out  = sycl::malloc_device<std::uint32_t>(t2_count, q);
    auto* d_t2_mi_out    = sycl::malloc_device<std::uint32_t>(t2_count, q);
    auto* d_t2_meta_out  = sycl::malloc_device<std::uint64_t>(t2_count, q);
    auto* d_t2_xbits_out = sycl::malloc_device<std::uint32_t>(t2_count, q);

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
        d_t2_mi, d_t2_mi_out, d_t2_idx_in, d_t2_idx_out,
        t2_count, 0, k, q);
    pos2gpu::launch_permute_t2(
        d_t2_meta, d_t2_xbits, d_t2_idx_out,
        d_t2_meta_out, d_t2_xbits_out, t2_count, q);
    q.wait();
    if (d_t2_sort_scratch) sycl::free(d_t2_sort_scratch, q);
    sycl::free(d_t2_idx_in,  q);
    sycl::free(d_t2_idx_out, q);
    sycl::free(d_t2_mi,      q);
    sycl::free(d_t2_meta,    q);
    sycl::free(d_t2_xbits,   q);

    T2Sorted out;
    out.mi.resize(t2_count);
    out.meta.resize(t2_count);
    out.xbits.resize(t2_count);
    if (t2_count > 0) {
        q.memcpy(out.mi.data(),    d_t2_mi_out,
                 t2_count * sizeof(std::uint32_t)).wait();
        q.memcpy(out.meta.data(),  d_t2_meta_out,
                 t2_count * sizeof(std::uint64_t)).wait();
        q.memcpy(out.xbits.data(), d_t2_xbits_out,
                 t2_count * sizeof(std::uint32_t)).wait();
    }
    sycl::free(d_t2_mi_out,    q);
    sycl::free(d_t2_meta_out,  q);
    sycl::free(d_t2_xbits_out, q);
    return out;
}

T2Sorted sharded_t2_phase(pos2gpu::BatchEntry const& entry)
{
    auto& q = pos2gpu::sycl_backend::queue();
    pos2gpu::BatchOptions opts{};
    opts.shard_plot = true;

    std::vector<pos2gpu::MultiGpuShardContext> shards(2);
    shards[0] = {&q, 0};
    shards[1] = {&q, 0};

    pos2gpu::MultiGpuPlotPipeline pipeline(entry, opts, std::move(shards));
    pipeline.run_through(pos2gpu::Phase::T2);

    std::uint64_t total = 0;
    for (std::size_t s = 0; s < pipeline.shard_count(); ++s) {
        total += pipeline.t2_phase_count(s);
    }

    T2Sorted out;
    out.mi.resize(total);
    out.meta.resize(total);
    out.xbits.resize(total);

    std::uint64_t off = 0;
    for (std::size_t s = 0; s < pipeline.shard_count(); ++s) {
        std::uint64_t const c = pipeline.t2_phase_count(s);
        if (c == 0) continue;
        sycl::queue& sq = pipeline.shard_queue(s);
        sq.memcpy(out.mi.data() + off,    pipeline.t2_phase_d_mi(s),
                  c * sizeof(std::uint32_t)).wait();
        sq.memcpy(out.meta.data() + off,  pipeline.t2_phase_d_meta(s),
                  c * sizeof(std::uint64_t)).wait();
        sq.memcpy(out.xbits.data() + off, pipeline.t2_phase_d_xbits(s),
                  c * sizeof(std::uint32_t)).wait();
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

    auto ref     = single_gpu_t2_phase(entry);
    auto sharded = sharded_t2_phase  (entry);

    // SET comparison: sort both by the full (mi, meta, xbits) tuple
    // before memcmp. T2 emits via atomic cursor so tie order is non-
    // deterministic between the reference and sharded paths.
    auto canonicalize = [](T2Sorted& s) {
        std::vector<std::uint64_t> idx(s.mi.size());
        for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
            [&](std::uint64_t a, std::uint64_t b) {
                if (s.mi   [a] != s.mi   [b]) return s.mi   [a] < s.mi   [b];
                if (s.meta [a] != s.meta [b]) return s.meta [a] < s.meta [b];
                return s.xbits[a] < s.xbits[b];
            });
        std::vector<std::uint32_t> mi2  (s.mi.size());
        std::vector<std::uint64_t> meta2(s.meta.size());
        std::vector<std::uint32_t> xb2  (s.xbits.size());
        for (std::size_t i = 0; i < idx.size(); ++i) {
            mi2  [i] = s.mi   [idx[i]];
            meta2[i] = s.meta [idx[i]];
            xb2  [i] = s.xbits[idx[i]];
        }
        s.mi    = std::move(mi2);
        s.meta  = std::move(meta2);
        s.xbits = std::move(xb2);
    };
    canonicalize(ref);
    canonicalize(sharded);

    bool const size_ok =
        (ref.mi.size()    == sharded.mi.size()) &&
        (ref.meta.size()  == sharded.meta.size()) &&
        (ref.xbits.size() == sharded.xbits.size());
    bool const mi_ok =
        size_ok && std::memcmp(
            ref.mi.data(), sharded.mi.data(),
            sizeof(std::uint32_t) * ref.mi.size()) == 0;
    bool const meta_ok =
        size_ok && std::memcmp(
            ref.meta.data(), sharded.meta.data(),
            sizeof(std::uint64_t) * ref.meta.size()) == 0;
    bool const xbits_ok =
        size_ok && std::memcmp(
            ref.xbits.data(), sharded.xbits.data(),
            sizeof(std::uint32_t) * ref.xbits.size()) == 0;
    bool const ok = size_ok && mi_ok && meta_ok && xbits_ok;

    std::printf(
        "%s t2-phase k=%d testnet=%d seed=%u  "
        "[count=%llu vs %llu  size=%d mi=%d meta=%d xbits=%d]\n",
        ok ? "PASS" : "FAIL", k, testnet ? 1 : 0,
        static_cast<unsigned>(plot_id_seed),
        static_cast<unsigned long long>(ref.mi.size()),
        static_cast<unsigned long long>(sharded.mi.size()),
        size_ok ? 1 : 0, mi_ok ? 1 : 0, meta_ok ? 1 : 0, xbits_ok ? 1 : 0);
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
