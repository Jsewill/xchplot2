// sycl_sharded_reference.hpp — shared single-GPU reference helpers for
// the sharded parity tests.
//
// The four sycl_t*_phase_sharded_parity / sycl_fragment_phase_sharded_-
// parity binaries each need a "what does the single-GPU pipeline
// produce up to phase X" function as their reference. Before this
// header, each binary inlined the full Xs + T1 + ... + X kernel chain
// — ~400 lines of duplication across four files. This header
// centralises the chain so a future change to a phase's kernel
// surface only has to be reflected in one place.
//
// Each function:
//   - Takes a BatchEntry by reference (k, strength, plot_id, testnet).
//   - Internally drives the full kernel chain on a single SYCL queue
//     (the singleton sycl_backend::queue()).
//   - Returns the end-of-phase output as host-side std::vector(s),
//     ready to be set-compared against the sharded path's per-shard
//     outputs concatenated in shard-id order.
//
// All ownership is handled internally; the host vectors returned to
// the caller are the only persistent allocations.

#pragma once

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
#include "host/PoolSizing.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace pos2gpu::parity {

// ---------------------------------------------------------------------
// Tiny RAII for device-side intermediates within these helpers. Local
// duplicate of the SyclDevicePtr / SyclDeviceVoid pattern in
// MultiGpuPlotPipeline.cpp so the parity-test header doesn't depend on
// pipeline internals.
// ---------------------------------------------------------------------
template <class T>
struct RaiiDev {
    sycl::queue* q = nullptr;
    T* p = nullptr;
    RaiiDev() = default;
    RaiiDev(sycl::queue* q_, T* p_) : q(q_), p(p_) {}
    RaiiDev(RaiiDev const&) = delete;
    RaiiDev& operator=(RaiiDev const&) = delete;
    RaiiDev(RaiiDev&& o) noexcept : q(o.q), p(o.p) { o.p = nullptr; }
    RaiiDev& operator=(RaiiDev&& o) noexcept {
        if (this != &o) { reset(); q = o.q; p = o.p; o.p = nullptr; }
        return *this;
    }
    ~RaiiDev() { reset(); }
    void reset() noexcept { if (p && q) { sycl::free(p, *q); p = nullptr; } }
    T* get() const noexcept { return p; }
    T* release() noexcept { T* tmp = p; p = nullptr; return tmp; }
};
template <class T>
inline RaiiDev<T> alloc_dev(std::size_t n, sycl::queue& q) {
    return RaiiDev<T>(&q, sycl::malloc_device<T>(n, q));
}
struct RaiiDevVoid {
    sycl::queue* q = nullptr;
    void* p = nullptr;
    RaiiDevVoid() = default;
    RaiiDevVoid(sycl::queue* q_, void* p_) : q(q_), p(p_) {}
    RaiiDevVoid(RaiiDevVoid const&) = delete;
    RaiiDevVoid& operator=(RaiiDevVoid const&) = delete;
    RaiiDevVoid(RaiiDevVoid&& o) noexcept : q(o.q), p(o.p) { o.p = nullptr; }
    RaiiDevVoid& operator=(RaiiDevVoid&& o) noexcept {
        if (this != &o) { reset(); q = o.q; p = o.p; o.p = nullptr; }
        return *this;
    }
    ~RaiiDevVoid() { reset(); }
    void reset() noexcept { if (p && q) { sycl::free(p, *q); p = nullptr; } }
    void* get() const noexcept { return p; }
};

// ---------------------------------------------------------------------
// Phase-output structs (host-side).
// ---------------------------------------------------------------------
struct T1Sorted {
    std::vector<std::uint32_t> mi;
    std::vector<std::uint64_t> meta;
};

struct T2Sorted {
    std::vector<std::uint32_t> mi;
    std::vector<std::uint64_t> meta;
    std::vector<std::uint32_t> xbits;
};

// ---------------------------------------------------------------------
// Helpers that return device-side intermediate state. Each runs one
// phase given the previous phase's device output, producing the next
// phase's output on device. Caller chains them; the returned RaiiDev
// owners free device buffers when they go out of scope.
// ---------------------------------------------------------------------
inline RaiiDev<XsCandidateGpu> run_xs_on_device(
    BatchEntry const& entry, sycl::queue& q)
{
    int const k = entry.k;
    std::uint64_t const total_xs = std::uint64_t{1} << k;
    std::uint32_t const xor_const = entry.testnet ? 0xA3B1C4D7u : 0u;
    AesHashKeys const keys = make_keys(entry.plot_id.data());

    auto d_keys_a = alloc_dev<std::uint32_t>(total_xs, q);
    auto d_vals_a = alloc_dev<std::uint32_t>(total_xs, q);
    auto d_keys_b = alloc_dev<std::uint32_t>(total_xs, q);
    auto d_vals_b = alloc_dev<std::uint32_t>(total_xs, q);
    auto d_xs     = alloc_dev<XsCandidateGpu>(total_xs, q);

    launch_xs_gen(keys, d_keys_a.get(), d_vals_a.get(), total_xs, k, xor_const, q);

    std::size_t sb = 0;
    launch_sort_pairs_u32_u32(
        nullptr, sb, nullptr, nullptr, nullptr, nullptr,
        total_xs, 0, k, q);
    RaiiDevVoid d_sc;
    if (sb) d_sc = RaiiDevVoid(&q, sycl::malloc_device(sb, q));
    launch_sort_pairs_u32_u32(
        d_sc.get() ? d_sc.get() : reinterpret_cast<void*>(std::uintptr_t{1}),
        sb,
        d_keys_a.get(), d_keys_b.get(), d_vals_a.get(), d_vals_b.get(),
        total_xs, 0, k, q);
    launch_xs_pack(d_keys_b.get(), d_vals_b.get(), d_xs.get(), total_xs, q);
    q.wait();

    return d_xs;
}

struct T1DeviceSorted {
    RaiiDev<std::uint32_t> d_mi;
    RaiiDev<std::uint64_t> d_meta;
    std::uint64_t          count = 0;
};

inline T1DeviceSorted run_t1_on_device(
    BatchEntry const& entry,
    XsCandidateGpu const* d_xs, std::uint64_t total_xs,
    sycl::queue& q)
{
    int const k = entry.k;
    auto t1p = make_t1_params(k, entry.strength);
    std::uint64_t const cap = match_phase_capacity(k, t1p.num_section_bits);

    auto d_t1_meta  = alloc_dev<std::uint64_t>(cap, q);
    auto d_t1_mi    = alloc_dev<std::uint32_t>(cap, q);
    auto d_t1_count = alloc_dev<std::uint64_t>(1,   q);

    std::size_t tb = 0;
    launch_t1_match(
        entry.plot_id.data(), t1p, d_xs, total_xs,
        nullptr, nullptr, nullptr, cap, nullptr, &tb, q);
    RaiiDevVoid d_temp;
    if (tb) d_temp = RaiiDevVoid(&q, sycl::malloc_device(tb, q));
    launch_t1_match(
        entry.plot_id.data(), t1p, d_xs, total_xs,
        d_t1_meta.get(), d_t1_mi.get(), d_t1_count.get(), cap,
        d_temp.get() ? d_temp.get() : reinterpret_cast<void*>(std::uintptr_t{1}),
        &tb, q);
    q.wait();

    std::uint64_t t1_count = 0;
    q.memcpy(&t1_count, d_t1_count.get(), sizeof(std::uint64_t)).wait();

    auto d_idx_in   = alloc_dev<std::uint32_t>(t1_count, q);
    auto d_idx_out  = alloc_dev<std::uint32_t>(t1_count, q);
    auto d_mi_s     = alloc_dev<std::uint32_t>(t1_count, q);
    auto d_meta_s   = alloc_dev<std::uint64_t>(t1_count, q);

    std::size_t sb = 0;
    launch_sort_pairs_u32_u32(
        nullptr, sb, nullptr, nullptr, nullptr, nullptr,
        t1_count, 0, k, q);
    RaiiDevVoid d_sc;
    if (sb) d_sc = RaiiDevVoid(&q, sycl::malloc_device(sb, q));
    launch_init_u32_identity(d_idx_in.get(), t1_count, q);
    launch_sort_pairs_u32_u32(
        d_sc.get() ? d_sc.get() : reinterpret_cast<void*>(std::uintptr_t{1}),
        sb,
        d_t1_mi.get(), d_mi_s.get(), d_idx_in.get(), d_idx_out.get(),
        t1_count, 0, k, q);
    launch_gather_u64(d_t1_meta.get(), d_idx_out.get(),
                      d_meta_s.get(), t1_count, q);
    q.wait();

    T1DeviceSorted out;
    out.d_mi   = std::move(d_mi_s);
    out.d_meta = std::move(d_meta_s);
    out.count  = t1_count;
    return out;
}

struct T2DeviceSorted {
    RaiiDev<std::uint32_t> d_mi;
    RaiiDev<std::uint64_t> d_meta;
    RaiiDev<std::uint32_t> d_xbits;
    std::uint64_t          count = 0;
};

inline T2DeviceSorted run_t2_on_device(
    BatchEntry const& entry,
    std::uint64_t const* d_t1_meta, std::uint32_t const* d_t1_mi,
    std::uint64_t t1_count, sycl::queue& q)
{
    int const k = entry.k;
    auto t2p = make_t2_params(k, entry.strength);
    std::uint64_t const cap = match_phase_capacity(k, t2p.num_section_bits);

    auto d_t2_meta  = alloc_dev<std::uint64_t>(cap, q);
    auto d_t2_mi    = alloc_dev<std::uint32_t>(cap, q);
    auto d_t2_xbits = alloc_dev<std::uint32_t>(cap, q);
    auto d_t2_count = alloc_dev<std::uint64_t>(1,   q);

    std::size_t tb = 0;
    launch_t2_match(
        entry.plot_id.data(), t2p, nullptr, nullptr, t1_count,
        nullptr, nullptr, nullptr, d_t2_count.get(), cap, nullptr, &tb, q);
    RaiiDevVoid d_temp;
    if (tb) d_temp = RaiiDevVoid(&q, sycl::malloc_device(tb, q));
    launch_t2_match(
        entry.plot_id.data(), t2p, d_t1_meta, d_t1_mi, t1_count,
        d_t2_meta.get(), d_t2_mi.get(), d_t2_xbits.get(),
        d_t2_count.get(), cap,
        d_temp.get() ? d_temp.get() : reinterpret_cast<void*>(std::uintptr_t{1}),
        &tb, q);
    q.wait();

    std::uint64_t t2_count = 0;
    q.memcpy(&t2_count, d_t2_count.get(), sizeof(std::uint64_t)).wait();

    auto d_idx_in    = alloc_dev<std::uint32_t>(t2_count, q);
    auto d_idx_out   = alloc_dev<std::uint32_t>(t2_count, q);
    auto d_mi_s      = alloc_dev<std::uint32_t>(t2_count, q);
    auto d_meta_s    = alloc_dev<std::uint64_t>(t2_count, q);
    auto d_xbits_s   = alloc_dev<std::uint32_t>(t2_count, q);

    std::size_t sb = 0;
    launch_sort_pairs_u32_u32(
        nullptr, sb, nullptr, nullptr, nullptr, nullptr,
        t2_count, 0, k, q);
    RaiiDevVoid d_sc;
    if (sb) d_sc = RaiiDevVoid(&q, sycl::malloc_device(sb, q));
    launch_init_u32_identity(d_idx_in.get(), t2_count, q);
    launch_sort_pairs_u32_u32(
        d_sc.get() ? d_sc.get() : reinterpret_cast<void*>(std::uintptr_t{1}),
        sb,
        d_t2_mi.get(), d_mi_s.get(), d_idx_in.get(), d_idx_out.get(),
        t2_count, 0, k, q);
    launch_permute_t2(
        d_t2_meta.get(), d_t2_xbits.get(), d_idx_out.get(),
        d_meta_s.get(), d_xbits_s.get(), t2_count, q);
    q.wait();

    T2DeviceSorted out;
    out.d_mi    = std::move(d_mi_s);
    out.d_meta  = std::move(d_meta_s);
    out.d_xbits = std::move(d_xbits_s);
    out.count   = t2_count;
    return out;
}

// T3 match — returns unsorted u64 fragments + count on device.
struct T3DeviceUnsorted {
    RaiiDev<std::uint64_t> d_frags;
    std::uint64_t          count = 0;
};

inline T3DeviceUnsorted run_t3_on_device(
    BatchEntry const& entry,
    std::uint64_t const* d_t2_meta,
    std::uint32_t const* d_t2_xbits,
    std::uint32_t const* d_t2_mi,
    std::uint64_t t2_count, sycl::queue& q)
{
    int const k = entry.k;
    auto t3p = make_t3_params(k, entry.strength);
    std::uint64_t const cap = match_phase_capacity(k, t3p.num_section_bits);

    auto d_t3       = alloc_dev<T3PairingGpu>(cap, q);
    auto d_t3_count = alloc_dev<std::uint64_t>(1,  q);

    std::size_t tb = 0;
    launch_t3_match(
        entry.plot_id.data(), t3p, nullptr, nullptr, nullptr, t2_count,
        nullptr, d_t3_count.get(), cap, nullptr, &tb, q);
    RaiiDevVoid d_temp;
    if (tb) d_temp = RaiiDevVoid(&q, sycl::malloc_device(tb, q));
    launch_t3_match(
        entry.plot_id.data(), t3p,
        d_t2_meta, d_t2_xbits, d_t2_mi, t2_count,
        d_t3.get(), d_t3_count.get(), cap,
        d_temp.get() ? d_temp.get() : reinterpret_cast<void*>(std::uintptr_t{1}),
        &tb, q);
    q.wait();

    std::uint64_t t3_count = 0;
    q.memcpy(&t3_count, d_t3_count.get(), sizeof(std::uint64_t)).wait();

    // Reinterpret the T3PairingGpu* as u64* — they're layout-identical.
    T3DeviceUnsorted out;
    out.d_frags = RaiiDev<std::uint64_t>(
        &q, reinterpret_cast<std::uint64_t*>(d_t3.release()));
    out.count   = t3_count;
    return out;
}

// ---------------------------------------------------------------------
// End-of-phase helpers that return host vectors. Each composes the
// device chain above and copies the final phase output to host.
// ---------------------------------------------------------------------
inline std::vector<XsCandidateGpu> single_gpu_xs(BatchEntry const& entry)
{
    auto& q = sycl_backend::queue();
    auto d_xs = run_xs_on_device(entry, q);
    std::uint64_t const total_xs = std::uint64_t{1} << entry.k;
    std::vector<XsCandidateGpu> out(total_xs);
    q.memcpy(out.data(), d_xs.get(),
             total_xs * sizeof(XsCandidateGpu)).wait();
    return out;
}

inline T1Sorted single_gpu_t1(BatchEntry const& entry)
{
    auto& q = sycl_backend::queue();
    auto d_xs = run_xs_on_device(entry, q);
    std::uint64_t const total_xs = std::uint64_t{1} << entry.k;
    auto t1 = run_t1_on_device(entry, d_xs.get(), total_xs, q);

    T1Sorted out;
    out.mi.resize(t1.count);
    out.meta.resize(t1.count);
    if (t1.count > 0) {
        q.memcpy(out.mi.data(),   t1.d_mi.get(),
                 t1.count * sizeof(std::uint32_t)).wait();
        q.memcpy(out.meta.data(), t1.d_meta.get(),
                 t1.count * sizeof(std::uint64_t)).wait();
    }
    return out;
}

inline T2Sorted single_gpu_t2(BatchEntry const& entry)
{
    auto& q = sycl_backend::queue();
    auto d_xs = run_xs_on_device(entry, q);
    std::uint64_t const total_xs = std::uint64_t{1} << entry.k;
    auto t1 = run_t1_on_device(entry, d_xs.get(), total_xs, q);
    auto t2 = run_t2_on_device(entry,
        t1.d_meta.get(), t1.d_mi.get(), t1.count, q);

    T2Sorted out;
    out.mi.resize(t2.count);
    out.meta.resize(t2.count);
    out.xbits.resize(t2.count);
    if (t2.count > 0) {
        q.memcpy(out.mi.data(),    t2.d_mi.get(),
                 t2.count * sizeof(std::uint32_t)).wait();
        q.memcpy(out.meta.data(),  t2.d_meta.get(),
                 t2.count * sizeof(std::uint64_t)).wait();
        q.memcpy(out.xbits.data(), t2.d_xbits.get(),
                 t2.count * sizeof(std::uint32_t)).wait();
    }
    return out;
}

// Returns T3 fragments AS EMITTED — not sorted. Used by the T3 phase
// parity test which compares as a multiset.
inline std::vector<std::uint64_t> single_gpu_t3_unsorted(BatchEntry const& entry)
{
    auto& q = sycl_backend::queue();
    auto d_xs = run_xs_on_device(entry, q);
    std::uint64_t const total_xs = std::uint64_t{1} << entry.k;
    auto t1 = run_t1_on_device(entry, d_xs.get(), total_xs, q);
    auto t2 = run_t2_on_device(entry,
        t1.d_meta.get(), t1.d_mi.get(), t1.count, q);
    auto t3 = run_t3_on_device(entry,
        t2.d_meta.get(), t2.d_xbits.get(), t2.d_mi.get(), t2.count, q);

    std::vector<std::uint64_t> out(t3.count);
    if (t3.count > 0) {
        q.memcpy(out.data(), t3.d_frags.get(),
                 t3.count * sizeof(std::uint64_t)).wait();
    }
    return out;
}

// Returns T3 fragments after sort over low 2k bits. Used by the
// T3-phase and fragment-phase parity tests.
inline std::vector<std::uint64_t> single_gpu_fragments(BatchEntry const& entry)
{
    auto& q = sycl_backend::queue();
    auto d_xs = run_xs_on_device(entry, q);
    std::uint64_t const total_xs = std::uint64_t{1} << entry.k;
    auto t1 = run_t1_on_device(entry, d_xs.get(), total_xs, q);
    auto t2 = run_t2_on_device(entry,
        t1.d_meta.get(), t1.d_mi.get(), t1.count, q);
    auto t3 = run_t3_on_device(entry,
        t2.d_meta.get(), t2.d_xbits.get(), t2.d_mi.get(), t2.count, q);

    auto d_frags_sorted = alloc_dev<std::uint64_t>(t3.count, q);
    std::size_t sb = 0;
    launch_sort_keys_u64(
        nullptr, sb, nullptr, nullptr,
        t3.count, 0, 2 * entry.k, q);
    RaiiDevVoid d_sc;
    if (sb) d_sc = RaiiDevVoid(&q, sycl::malloc_device(sb, q));
    launch_sort_keys_u64(
        d_sc.get() ? d_sc.get() : reinterpret_cast<void*>(std::uintptr_t{1}),
        sb,
        t3.d_frags.get(), d_frags_sorted.get(),
        t3.count, 0, 2 * entry.k, q);
    q.wait();

    std::vector<std::uint64_t> out(t3.count);
    if (t3.count > 0) {
        q.memcpy(out.data(), d_frags_sorted.get(),
                 t3.count * sizeof(std::uint64_t)).wait();
    }
    return out;
}

} // namespace pos2gpu::parity
