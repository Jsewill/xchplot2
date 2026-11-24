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

using pos2gpu::parity::T2Sorted;

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

    auto ref     = pos2gpu::parity::single_gpu_t2(entry);
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
        for (bool testnet : {false}) {
            for (std::uint8_t seed : {7u, 31u}) {
                all_ok = run_one(k, testnet, seed) && all_ok;
            }
        }
    }
    return all_ok ? 0 : 1;
}
