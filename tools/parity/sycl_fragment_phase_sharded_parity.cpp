// sycl_fragment_phase_sharded_parity — drives MultiGpuPlotPipeline's
// run() (Phase 2.3d) end-to-end and verifies the host-bounced
// concatenated proof_fragment span is multiset-equivalent to the
// single-GPU sorted output.
//
// The T3 phase parity test already verifies the per-shard sorted
// fragments match single-GPU as a multiset. This test additionally
// validates the D2H + concat in run_fragment_phase: per-shard counts
// summing to total, contiguous host buffer landing, no off-by-one in
// the bucket-aligned concatenation.
//
// Comparison is by full u64 fragment value (sort + memcmp) — same
// reasoning as the T3 phase parity test (radix sort over only the
// low 2k bits leaves the high bits arbitrary).

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

std::vector<std::uint64_t> sharded_fragments(
    pos2gpu::BatchEntry const& entry,
    double w0 = 1.0, double w1 = 1.0,
    bool prefer_peer = false)
{
    auto& q = pos2gpu::sycl_backend::queue();
    pos2gpu::BatchOptions opts{};
    opts.shard_plot       = true;
    opts.prefer_peer_copy = prefer_peer;

    std::vector<pos2gpu::MultiGpuShardContext> shards(2);
    shards[0] = {&q, 0, w0};
    shards[1] = {&q, 0, w1};

    pos2gpu::MultiGpuPlotPipeline pipeline(entry, opts, std::move(shards));
    pipeline.run();  // through fragment phase

    auto span = pipeline.fragments();
    return std::vector<std::uint64_t>(span.begin(), span.end());
}

bool run_one(int k, bool testnet, std::uint8_t plot_id_seed,
             double w0 = 1.0, double w1 = 1.0,
             bool prefer_peer = false)
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
    auto sharded = sharded_fragments  (entry, w0, w1, prefer_peer);

    std::sort(ref.begin(),     ref.end());
    std::sort(sharded.begin(), sharded.end());

    bool const size_ok  = (ref.size() == sharded.size());
    bool const bytes_ok = size_ok && std::memcmp(
        ref.data(), sharded.data(),
        sizeof(std::uint64_t) * ref.size()) == 0;
    bool const ok = size_ok && bytes_ok;

    std::printf(
        "%s fragment-phase k=%d testnet=%d seed=%u w=[%g,%g] xport=%s  "
        "[count=%llu vs %llu  size=%d bytes=%d]\n",
        ok ? "PASS" : "FAIL", k, testnet ? 1 : 0,
        static_cast<unsigned>(plot_id_seed), w0, w1,
        prefer_peer ? "peer" : "host",
        static_cast<unsigned long long>(ref.size()),
        static_cast<unsigned long long>(sharded.size()),
        size_ok ? 1 : 0, bytes_ok ? 1 : 0);
    return ok;
}

} // namespace

int main()
{
    bool all_ok = true;
    // HostBounce + uniform weights — Phase 2.3d coverage.
    for (int k : {18, 20, 22}) {
        for (bool testnet : {false}) {
            for (std::uint8_t seed : {7u, 31u}) {
                all_ok = run_one(k, testnet, seed) && all_ok;
            }
        }
    }
    // HostBounce + weighted (Phase 2.4a).
    for (int k : {18, 22}) {
        for (bool testnet : {false}) {
            for (std::uint8_t seed : {7u}) {
                all_ok = run_one(k, testnet, seed, /*w0=*/3.0, /*w1=*/1.0) && all_ok;
                all_ok = run_one(k, testnet, seed, /*w0=*/1.0, /*w1=*/3.0) && all_ok;
            }
        }
    }
    // Peer transport (Phase 2.4b). Default weights and the [3,1] skew.
    // Atomic-scatter on GPU produces non-deterministic tie order, so
    // multiset equivalence is the only correctness invariant — already
    // what this test checks (sort + memcmp).
    for (int k : {18, 20, 22}) {
        for (bool testnet : {false}) {
            for (std::uint8_t seed : {7u, 31u}) {
                all_ok = run_one(k, testnet, seed,
                                 /*w0=*/1.0, /*w1=*/1.0,
                                 /*prefer_peer=*/true) && all_ok;
            }
        }
    }
    for (int k : {18, 22}) {
        all_ok = run_one(k, /*testnet=*/false, /*seed=*/7u,
                         /*w0=*/3.0, /*w1=*/1.0,
                         /*prefer_peer=*/true) && all_ok;
    }
    return all_ok ? 0 : 1;
}
