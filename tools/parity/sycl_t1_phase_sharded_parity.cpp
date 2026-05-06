// sycl_t1_phase_sharded_parity — exercises MultiGpuPlotPipeline's T1
// phase (Phase 2.3a) with N=2 virtual shards on the same physical queue
// and verifies the union of per-shard sorted (mi, meta) outputs is
// byte-identical to the single-GPU T1 phase output (launch_xs_gen +
// xs sort + xs pack + launch_t1_match + identity-index radix sort by
// mi + 64-bit meta gather).
//
// Validates the full Phase 2.3a stack on a 1-GPU dev box: replicate
// sorted Xs across shards, per-shard launch_t1_match_prepare +
// launch_t1_match_range over the shard's bucket subset, and the new
// launch_sort_pairs_u32_u64_distributed bouncing T1 (mi, meta) across
// the host-pinned bridge. Real multi-physical-GPU runs are downstream.

#include "gpu/AesHashGpu.cuh"
#include "gpu/PipelineKernels.cuh"
#include "gpu/Sort.cuh"
#include "gpu/SyclBackend.hpp"
#include "gpu/T1Kernel.cuh"
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

struct T1Sorted {
    std::vector<std::uint32_t> mi;
    std::vector<std::uint64_t> meta;
};

// Reference: single-GPU Xs phase, then single-GPU T1 match, then T1
// sort by mi (identity-index radix sort + 64-bit meta gather), mirror-
// ing GpuPipeline.cpp's T1 path. The result is the sorted (mi, meta)
// streams that the production pipeline feeds into T2.
T1Sorted single_gpu_t1_phase(pos2gpu::BatchEntry const& entry)
{
    auto& q = pos2gpu::sycl_backend::queue();
    int const k = entry.k;
    std::uint64_t const total_xs = std::uint64_t{1} << k;
    std::uint32_t const xor_const = entry.testnet ? 0xA3B1C4D7u : 0u;
    pos2gpu::AesHashKeys const keys =
        pos2gpu::make_keys(entry.plot_id.data());

    // ----- Xs phase: gen + sort + pack into XsCandidateGpu. -----
    std::uint32_t* d_keys_a = sycl::malloc_device<std::uint32_t>(total_xs, q);
    std::uint32_t* d_vals_a = sycl::malloc_device<std::uint32_t>(total_xs, q);
    std::uint32_t* d_keys_b = sycl::malloc_device<std::uint32_t>(total_xs, q);
    std::uint32_t* d_vals_b = sycl::malloc_device<std::uint32_t>(total_xs, q);
    pos2gpu::XsCandidateGpu* d_xs =
        sycl::malloc_device<pos2gpu::XsCandidateGpu>(total_xs, q);

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
    sycl::free(d_keys_a, q);
    sycl::free(d_vals_a, q);
    sycl::free(d_keys_b, q);
    sycl::free(d_vals_b, q);

    // ----- T1 match. -----
    auto t1p = pos2gpu::make_t1_params(k, entry.strength);
    std::uint64_t const cap =
        pos2gpu::match_phase_capacity(k, t1p.num_section_bits);
    auto* d_t1_meta  = sycl::malloc_device<std::uint64_t>(cap, q);
    auto* d_t1_mi    = sycl::malloc_device<std::uint32_t>(cap, q);
    auto* d_t1_count = sycl::malloc_device<std::uint64_t>(1, q);

    std::size_t t1_temp_bytes = 0;
    pos2gpu::launch_t1_match(
        entry.plot_id.data(), t1p, d_xs, total_xs,
        nullptr, nullptr, nullptr, cap,
        nullptr, &t1_temp_bytes, q);
    void* d_t1_temp = t1_temp_bytes
        ? sycl::malloc_device(t1_temp_bytes, q) : nullptr;
    pos2gpu::launch_t1_match(
        entry.plot_id.data(), t1p, d_xs, total_xs,
        d_t1_meta, d_t1_mi, d_t1_count, cap,
        d_t1_temp ? d_t1_temp : reinterpret_cast<void*>(std::uintptr_t{1}),
        &t1_temp_bytes, q);
    q.wait();
    sycl::free(d_xs, q);
    if (d_t1_temp) sycl::free(d_t1_temp, q);

    std::uint64_t t1_count = 0;
    q.memcpy(&t1_count, d_t1_count, sizeof(std::uint64_t)).wait();
    sycl::free(d_t1_count, q);

    // ----- T1 sort by mi: identity-index radix sort + 64-bit gather. -----
    auto* d_idx_in  = sycl::malloc_device<std::uint32_t>(t1_count, q);
    auto* d_idx_out = sycl::malloc_device<std::uint32_t>(t1_count, q);
    auto* d_mi_out  = sycl::malloc_device<std::uint32_t>(t1_count, q);
    auto* d_meta_out = sycl::malloc_device<std::uint64_t>(t1_count, q);

    std::size_t sort_bytes = 0;
    pos2gpu::launch_sort_pairs_u32_u32(
        nullptr, sort_bytes, nullptr, nullptr, nullptr, nullptr,
        t1_count, 0, k, q);
    void* d_sort_scratch = sort_bytes
        ? sycl::malloc_device(sort_bytes, q) : nullptr;

    pos2gpu::launch_init_u32_identity(d_idx_in, t1_count, q);
    pos2gpu::launch_sort_pairs_u32_u32(
        d_sort_scratch ? d_sort_scratch
                       : reinterpret_cast<void*>(std::uintptr_t{1}),
        sort_bytes,
        d_t1_mi, d_mi_out, d_idx_in, d_idx_out,
        t1_count, 0, k, q);
    pos2gpu::launch_gather_u64(d_t1_meta, d_idx_out, d_meta_out, t1_count, q);
    q.wait();

    if (d_sort_scratch) sycl::free(d_sort_scratch, q);
    sycl::free(d_idx_in,  q);
    sycl::free(d_idx_out, q);
    sycl::free(d_t1_mi,   q);
    sycl::free(d_t1_meta, q);

    T1Sorted out;
    out.mi.resize(t1_count);
    out.meta.resize(t1_count);
    if (t1_count > 0) {
        q.memcpy(out.mi.data(),   d_mi_out,
                 t1_count * sizeof(std::uint32_t)).wait();
        q.memcpy(out.meta.data(), d_meta_out,
                 t1_count * sizeof(std::uint64_t)).wait();
    }
    sycl::free(d_mi_out,   q);
    sycl::free(d_meta_out, q);
    return out;
}

// Sharded: 2 virtual shards on the same queue, drive run_through(T1)
// and concatenate the per-shard sorted (mi, meta) outputs in shard-id
// order. By construction (distributed sort partitions by mi value-space)
// the concatenation reproduces a globally sorted-by-mi stream — i.e. the
// single-GPU reference.
T1Sorted sharded_t1_phase(pos2gpu::BatchEntry const& entry)
{
    auto& q = pos2gpu::sycl_backend::queue();
    pos2gpu::BatchOptions opts{};
    opts.shard_plot = true;

    std::vector<pos2gpu::MultiGpuShardContext> shards(2);
    shards[0] = {&q, 0};
    shards[1] = {&q, 0};

    pos2gpu::MultiGpuPlotPipeline pipeline(entry, opts, std::move(shards));
    pipeline.run_through(pos2gpu::Phase::T1);

    std::uint64_t total = 0;
    for (std::size_t s = 0; s < pipeline.shard_count(); ++s) {
        total += pipeline.t1_phase_count(s);
    }

    T1Sorted out;
    out.mi.resize(total);
    out.meta.resize(total);

    std::uint64_t off = 0;
    for (std::size_t s = 0; s < pipeline.shard_count(); ++s) {
        std::uint64_t const c = pipeline.t1_phase_count(s);
        if (c == 0) continue;
        sycl::queue& sq = pipeline.shard_queue(s);
        sq.memcpy(out.mi.data() + off, pipeline.t1_phase_d_mi(s),
                  c * sizeof(std::uint32_t)).wait();
        sq.memcpy(out.meta.data() + off, pipeline.t1_phase_d_meta(s),
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

    auto ref     = single_gpu_t1_phase(entry);
    auto sharded = sharded_t1_phase  (entry);

    // T1 emits matches via an atomic cursor — same-mi entries can land
    // in any order within either run. Compare as a SET by sorting both
    // (mi, meta) sequences lexicographically. This matches the way the
    // existing sycl_t1_parity test compares CPU vs GPU T1 outputs.
    auto canonicalize = [](T1Sorted& s) {
        std::vector<std::uint64_t> idx(s.mi.size());
        for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
            [&](std::uint64_t a, std::uint64_t b) {
                if (s.mi[a] != s.mi[b]) return s.mi[a] < s.mi[b];
                return s.meta[a] < s.meta[b];
            });
        std::vector<std::uint32_t> mi2(s.mi.size());
        std::vector<std::uint64_t> meta2(s.meta.size());
        for (std::size_t i = 0; i < idx.size(); ++i) {
            mi2[i]   = s.mi  [idx[i]];
            meta2[i] = s.meta[idx[i]];
        }
        s.mi   = std::move(mi2);
        s.meta = std::move(meta2);
    };
    canonicalize(ref);
    canonicalize(sharded);

    bool const size_ok =
        (ref.mi.size() == sharded.mi.size()) &&
        (ref.meta.size() == sharded.meta.size());
    bool const mi_ok =
        size_ok && std::memcmp(
            ref.mi.data(), sharded.mi.data(),
            sizeof(std::uint32_t) * ref.mi.size()) == 0;
    bool const meta_ok =
        size_ok && std::memcmp(
            ref.meta.data(), sharded.meta.data(),
            sizeof(std::uint64_t) * ref.meta.size()) == 0;
    bool const ok = size_ok && mi_ok && meta_ok;

    std::printf(
        "%s t1-phase k=%d testnet=%d seed=%u  "
        "[count=%llu vs %llu  size=%d mi=%d meta=%d]\n",
        ok ? "PASS" : "FAIL", k, testnet ? 1 : 0,
        static_cast<unsigned>(plot_id_seed),
        static_cast<unsigned long long>(ref.mi.size()),
        static_cast<unsigned long long>(sharded.mi.size()),
        size_ok ? 1 : 0, mi_ok ? 1 : 0, meta_ok ? 1 : 0);
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
