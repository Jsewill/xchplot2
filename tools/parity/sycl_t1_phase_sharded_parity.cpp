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

#include "gpu/SyclBackend.hpp"
#include "host/BatchPlotter.hpp"
#include "host/MultiGpuPlotPipeline.hpp"

#include "sycl_sharded_reference.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using pos2gpu::parity::T1Sorted;

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

    auto ref     = pos2gpu::parity::single_gpu_t1(entry);
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
