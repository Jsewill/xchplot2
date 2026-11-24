// sycl_t3_phase_sharded_parity — exercises MultiGpuPlotPipeline's T3
// phase (Phase 2.3c) with N=2 virtual shards on the same physical queue
// and verifies the union of per-shard sorted u64 proof_fragments is
// byte-identical (as a multiset) to the single-GPU T3 phase output.
//
// T3 emits T3PairingGpu = single u64 proof_fragment via atomic cursor;
// post-sort tie order isn't deterministic between the reference and
// sharded paths, so the comparison sorts both arrays before memcmp.

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

    auto ref     = pos2gpu::parity::single_gpu_fragments(entry);
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
        for (bool testnet : {false}) {
            for (std::uint8_t seed : {7u, 31u}) {
                all_ok = run_one(k, testnet, seed) && all_ok;
            }
        }
    }
    return all_ok ? 0 : 1;
}
