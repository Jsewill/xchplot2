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

#if 0
// Inline single-GPU reference replaced by pos2gpu::parity::single_gpu_fragments
// from sycl_sharded_reference.hpp.
std::vector<std::uint64_t> single_gpu_fragments_LEGACY(
    pos2gpu::BatchEntry const& entry)
{
    auto& q = pos2gpu::sycl_backend::queue();
    int const k = entry.k;
    std::uint64_t const total_xs = std::uint64_t{1} << k;
    std::uint32_t const xor_const = entry.testnet ? 0xA3B1C4D7u : 0u;
    pos2gpu::AesHashKeys const keys =
        pos2gpu::make_keys(entry.plot_id.data());

    std::uint32_t* d_keys_a = sycl::malloc_device<std::uint32_t>(total_xs, q);
    std::uint32_t* d_vals_a = sycl::malloc_device<std::uint32_t>(total_xs, q);
    std::uint32_t* d_keys_b = sycl::malloc_device<std::uint32_t>(total_xs, q);
    std::uint32_t* d_vals_b = sycl::malloc_device<std::uint32_t>(total_xs, q);
    auto* d_xs = sycl::malloc_device<pos2gpu::XsCandidateGpu>(total_xs, q);

    pos2gpu::launch_xs_gen(keys, d_keys_a, d_vals_a, total_xs, k, xor_const, q);

    std::size_t xs_sb = 0;
    pos2gpu::launch_sort_pairs_u32_u32(
        nullptr, xs_sb, nullptr, nullptr, nullptr, nullptr,
        total_xs, 0, k, q);
    void* d_xs_sc = xs_sb ? sycl::malloc_device(xs_sb, q) : nullptr;
    pos2gpu::launch_sort_pairs_u32_u32(
        d_xs_sc ? d_xs_sc : reinterpret_cast<void*>(std::uintptr_t{1}),
        xs_sb, d_keys_a, d_keys_b, d_vals_a, d_vals_b,
        total_xs, 0, k, q);
    pos2gpu::launch_xs_pack(d_keys_b, d_vals_b, d_xs, total_xs, q);
    q.wait();
    if (d_xs_sc) sycl::free(d_xs_sc, q);
    sycl::free(d_keys_a, q); sycl::free(d_vals_a, q);
    sycl::free(d_keys_b, q); sycl::free(d_vals_b, q);

    auto t1p = pos2gpu::make_t1_params(k, entry.strength);
    std::uint64_t const t1_cap =
        pos2gpu::match_phase_capacity(k, t1p.num_section_bits);
    auto* d_t1_meta  = sycl::malloc_device<std::uint64_t>(t1_cap, q);
    auto* d_t1_mi    = sycl::malloc_device<std::uint32_t>(t1_cap, q);
    auto* d_t1_count = sycl::malloc_device<std::uint64_t>(1, q);

    std::size_t t1_tb = 0;
    pos2gpu::launch_t1_match(
        entry.plot_id.data(), t1p, d_xs, total_xs,
        nullptr, nullptr, nullptr, t1_cap, nullptr, &t1_tb, q);
    void* d_t1_temp = t1_tb ? sycl::malloc_device(t1_tb, q) : nullptr;
    pos2gpu::launch_t1_match(
        entry.plot_id.data(), t1p, d_xs, total_xs,
        d_t1_meta, d_t1_mi, d_t1_count, t1_cap,
        d_t1_temp ? d_t1_temp : reinterpret_cast<void*>(std::uintptr_t{1}),
        &t1_tb, q);
    q.wait();
    sycl::free(d_xs, q);
    if (d_t1_temp) sycl::free(d_t1_temp, q);

    std::uint64_t t1_count = 0;
    q.memcpy(&t1_count, d_t1_count, sizeof(std::uint64_t)).wait();
    sycl::free(d_t1_count, q);

    auto* d_t1_idx_in   = sycl::malloc_device<std::uint32_t>(t1_count, q);
    auto* d_t1_idx_out  = sycl::malloc_device<std::uint32_t>(t1_count, q);
    auto* d_t1_mi_s     = sycl::malloc_device<std::uint32_t>(t1_count, q);
    auto* d_t1_meta_s   = sycl::malloc_device<std::uint64_t>(t1_count, q);

    std::size_t t1_sb = 0;
    pos2gpu::launch_sort_pairs_u32_u32(
        nullptr, t1_sb, nullptr, nullptr, nullptr, nullptr,
        t1_count, 0, k, q);
    void* d_t1_sc = t1_sb ? sycl::malloc_device(t1_sb, q) : nullptr;
    pos2gpu::launch_init_u32_identity(d_t1_idx_in, t1_count, q);
    pos2gpu::launch_sort_pairs_u32_u32(
        d_t1_sc ? d_t1_sc : reinterpret_cast<void*>(std::uintptr_t{1}),
        t1_sb, d_t1_mi, d_t1_mi_s, d_t1_idx_in, d_t1_idx_out,
        t1_count, 0, k, q);
    pos2gpu::launch_gather_u64(
        d_t1_meta, d_t1_idx_out, d_t1_meta_s, t1_count, q);
    q.wait();
    if (d_t1_sc) sycl::free(d_t1_sc, q);
    sycl::free(d_t1_idx_in,  q);
    sycl::free(d_t1_idx_out, q);
    sycl::free(d_t1_mi,      q);
    sycl::free(d_t1_meta,    q);

    auto t2p = pos2gpu::make_t2_params(k, entry.strength);
    std::uint64_t const t2_cap =
        pos2gpu::match_phase_capacity(k, t2p.num_section_bits);
    auto* d_t2_meta  = sycl::malloc_device<std::uint64_t>(t2_cap, q);
    auto* d_t2_mi    = sycl::malloc_device<std::uint32_t>(t2_cap, q);
    auto* d_t2_xbits = sycl::malloc_device<std::uint32_t>(t2_cap, q);
    auto* d_t2_count = sycl::malloc_device<std::uint64_t>(1, q);

    std::size_t t2_tb = 0;
    pos2gpu::launch_t2_match(
        entry.plot_id.data(), t2p, nullptr, nullptr, t1_count,
        nullptr, nullptr, nullptr, d_t2_count, t2_cap, nullptr, &t2_tb, q);
    void* d_t2_temp = t2_tb ? sycl::malloc_device(t2_tb, q) : nullptr;
    pos2gpu::launch_t2_match(
        entry.plot_id.data(), t2p,
        d_t1_meta_s, d_t1_mi_s, t1_count,
        d_t2_meta, d_t2_mi, d_t2_xbits, d_t2_count, t2_cap,
        d_t2_temp ? d_t2_temp : reinterpret_cast<void*>(std::uintptr_t{1}),
        &t2_tb, q);
    q.wait();
    sycl::free(d_t1_mi_s, q);
    sycl::free(d_t1_meta_s, q);
    if (d_t2_temp) sycl::free(d_t2_temp, q);

    std::uint64_t t2_count = 0;
    q.memcpy(&t2_count, d_t2_count, sizeof(std::uint64_t)).wait();
    sycl::free(d_t2_count, q);

    auto* d_t2_idx_in   = sycl::malloc_device<std::uint32_t>(t2_count, q);
    auto* d_t2_idx_out  = sycl::malloc_device<std::uint32_t>(t2_count, q);
    auto* d_t2_mi_s     = sycl::malloc_device<std::uint32_t>(t2_count, q);
    auto* d_t2_meta_s   = sycl::malloc_device<std::uint64_t>(t2_count, q);
    auto* d_t2_xbits_s  = sycl::malloc_device<std::uint32_t>(t2_count, q);

    std::size_t t2_sb = 0;
    pos2gpu::launch_sort_pairs_u32_u32(
        nullptr, t2_sb, nullptr, nullptr, nullptr, nullptr,
        t2_count, 0, k, q);
    void* d_t2_sc = t2_sb ? sycl::malloc_device(t2_sb, q) : nullptr;
    pos2gpu::launch_init_u32_identity(d_t2_idx_in, t2_count, q);
    pos2gpu::launch_sort_pairs_u32_u32(
        d_t2_sc ? d_t2_sc : reinterpret_cast<void*>(std::uintptr_t{1}),
        t2_sb, d_t2_mi, d_t2_mi_s, d_t2_idx_in, d_t2_idx_out,
        t2_count, 0, k, q);
    pos2gpu::launch_permute_t2(
        d_t2_meta, d_t2_xbits, d_t2_idx_out,
        d_t2_meta_s, d_t2_xbits_s, t2_count, q);
    q.wait();
    if (d_t2_sc) sycl::free(d_t2_sc, q);
    sycl::free(d_t2_idx_in,  q);
    sycl::free(d_t2_idx_out, q);
    sycl::free(d_t2_mi,      q);
    sycl::free(d_t2_meta,    q);
    sycl::free(d_t2_xbits,   q);

    auto t3p = pos2gpu::make_t3_params(k, entry.strength);
    std::uint64_t const t3_cap =
        pos2gpu::match_phase_capacity(k, t3p.num_section_bits);
    auto* d_t3       = sycl::malloc_device<pos2gpu::T3PairingGpu>(t3_cap, q);
    auto* d_t3_count = sycl::malloc_device<std::uint64_t>(1, q);

    std::size_t t3_tb = 0;
    pos2gpu::launch_t3_match(
        entry.plot_id.data(), t3p,
        nullptr, nullptr, nullptr, t2_count,
        nullptr, d_t3_count, t3_cap, nullptr, &t3_tb, q);
    void* d_t3_temp = t3_tb ? sycl::malloc_device(t3_tb, q) : nullptr;
    pos2gpu::launch_t3_match(
        entry.plot_id.data(), t3p,
        d_t2_meta_s, d_t2_xbits_s, d_t2_mi_s, t2_count,
        d_t3, d_t3_count, t3_cap,
        d_t3_temp ? d_t3_temp : reinterpret_cast<void*>(std::uintptr_t{1}),
        &t3_tb, q);
    q.wait();
    sycl::free(d_t2_mi_s,    q);
    sycl::free(d_t2_meta_s,  q);
    sycl::free(d_t2_xbits_s, q);
    if (d_t3_temp) sycl::free(d_t3_temp, q);

    std::uint64_t t3_count = 0;
    q.memcpy(&t3_count, d_t3_count, sizeof(std::uint64_t)).wait();
    sycl::free(d_t3_count, q);

    auto* d_frags_in  = reinterpret_cast<std::uint64_t*>(d_t3);
    auto* d_frags_out = sycl::malloc_device<std::uint64_t>(t3_count, q);

    std::size_t t3_sb = 0;
    pos2gpu::launch_sort_keys_u64(
        nullptr, t3_sb, nullptr, nullptr,
        t3_count, 0, 2 * k, q);
    void* d_t3_sc = t3_sb ? sycl::malloc_device(t3_sb, q) : nullptr;
    pos2gpu::launch_sort_keys_u64(
        d_t3_sc ? d_t3_sc : reinterpret_cast<void*>(std::uintptr_t{1}),
        t3_sb, d_frags_in, d_frags_out, t3_count, 0, 2 * k, q);
    q.wait();
    if (d_t3_sc) sycl::free(d_t3_sc, q);

    std::vector<std::uint64_t> out(t3_count);
    if (t3_count > 0) {
        q.memcpy(out.data(), d_frags_out,
                 t3_count * sizeof(std::uint64_t)).wait();
    }
    sycl::free(d_t3,        q);
    sycl::free(d_frags_out, q);
    return out;
}
#endif

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
        for (bool testnet : {false, true}) {
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
        for (bool testnet : {false, true}) {
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
