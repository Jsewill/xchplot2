// SortDistributedPeer.cpp — peer-transport scatter kernels for the
// distributed sort variants. See SortDistributed.cpp for the entry
// points that branch between HostBounce and Peer.
//
// Atomic-scatter design: pass 1 counts items per destination via
// atomic increments; host computes per-destination start offsets;
// pass 2 atomically allocates a slot inside each destination's range
// and writes the item there. Tie order within a destination is
// non-deterministic (atomic completion order) but the multiset is
// preserved. Parity tests compare as a SET to absorb this.

#include "gpu/SortDistributedPeer.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace pos2gpu::detail {

namespace {

// Device-side bucket-of for u32 keys; mirrors the host bucket_of_u32
// formula in SortDistributed.cpp bit-for-bit.
inline std::size_t bucket_of_u32_dev(std::uint32_t key, std::uint32_t N,
                                     int begin_bit, int end_bit)
{
    int const bits = end_bit - begin_bit;
    if (bits >= 32) {
        return (static_cast<std::uint64_t>(key) *
                static_cast<std::uint64_t>(N)) >> 32;
    }
    std::uint32_t const mask  = (1u << bits) - 1u;
    std::uint32_t const value = (key >> begin_bit) & mask;
    return (static_cast<std::uint64_t>(value) *
            static_cast<std::uint64_t>(N)) >> bits;
}

inline std::size_t bucket_of_u64_dev(std::uint64_t key, std::uint32_t N,
                                     int begin_bit, int end_bit)
{
    int const bits = end_bit - begin_bit;
    if (bits >= 64) {
        std::uint32_t const hi = static_cast<std::uint32_t>(key >> 32);
        return (static_cast<std::uint64_t>(hi) *
                static_cast<std::uint64_t>(N)) >> 32;
    }
    if (bits > 32) {
        int const shift = begin_bit + (bits - 32);
        std::uint64_t const hi_mask = (1ull << (bits - 32)) - 1ull;
        std::uint64_t const value = (key >> shift) & hi_mask;
        return (value * static_cast<std::uint64_t>(N)) >> (bits - 32);
    }
    std::uint64_t const mask  = (1ull << bits) - 1ull;
    std::uint64_t const value = (key >> begin_bit) & mask;
    return (value * static_cast<std::uint64_t>(N)) >> bits;
}

} // namespace

void launch_count_per_dest_u32(
    std::uint32_t const* d_keys, std::uint64_t count,
    std::uint32_t N, int begin_bit, int end_bit,
    std::uint32_t* d_dst_count, sycl::queue& q)
{
    if (count == 0) return;
    constexpr std::size_t threads = 256;
    std::size_t const groups = (count + threads - 1) / threads;
    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=](sycl::nd_item<1> it) {
            std::uint64_t i = it.get_global_id(0);
            if (i >= count) return;
            std::uint32_t k = d_keys[i];
            std::size_t d = bucket_of_u32_dev(k, N, begin_bit, end_bit);
            sycl::atomic_ref<std::uint32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                cnt(d_dst_count[d]);
            cnt.fetch_add(1u);
        }).wait();
}

void launch_count_per_dest_u64(
    std::uint64_t const* d_keys, std::uint64_t count,
    std::uint32_t N, int begin_bit, int end_bit,
    std::uint32_t* d_dst_count, sycl::queue& q)
{
    if (count == 0) return;
    constexpr std::size_t threads = 256;
    std::size_t const groups = (count + threads - 1) / threads;
    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=](sycl::nd_item<1> it) {
            std::uint64_t i = it.get_global_id(0);
            if (i >= count) return;
            std::uint64_t k = d_keys[i];
            std::size_t d = bucket_of_u64_dev(k, N, begin_bit, end_bit);
            sycl::atomic_ref<std::uint32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                cnt(d_dst_count[d]);
            cnt.fetch_add(1u);
        }).wait();
}

void launch_scatter_u32_u32(
    std::uint32_t const* d_keys_in, std::uint32_t const* d_vals_in,
    std::uint64_t count, std::uint32_t N, int begin_bit, int end_bit,
    std::uint32_t const* d_dst_offsets, std::uint32_t* d_dst_cur,
    std::uint32_t* d_staging_keys, std::uint32_t* d_staging_vals,
    sycl::queue& q)
{
    if (count == 0) return;
    constexpr std::size_t threads = 256;
    std::size_t const groups = (count + threads - 1) / threads;
    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=](sycl::nd_item<1> it) {
            std::uint64_t i = it.get_global_id(0);
            if (i >= count) return;
            std::uint32_t const k = d_keys_in[i];
            std::uint32_t const v = d_vals_in[i];
            std::size_t const d = bucket_of_u32_dev(k, N, begin_bit, end_bit);
            sycl::atomic_ref<std::uint32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                cur(d_dst_cur[d]);
            std::uint32_t const slot = cur.fetch_add(1u);
            std::uint32_t const idx  = d_dst_offsets[d] + slot;
            d_staging_keys[idx] = k;
            d_staging_vals[idx] = v;
        }).wait();
}

void launch_scatter_u32_u64(
    std::uint32_t const* d_keys_in, std::uint64_t const* d_vals_in,
    std::uint64_t count, std::uint32_t N, int begin_bit, int end_bit,
    std::uint32_t const* d_dst_offsets, std::uint32_t* d_dst_cur,
    std::uint32_t* d_staging_keys, std::uint64_t* d_staging_vals,
    sycl::queue& q)
{
    if (count == 0) return;
    constexpr std::size_t threads = 256;
    std::size_t const groups = (count + threads - 1) / threads;
    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=](sycl::nd_item<1> it) {
            std::uint64_t i = it.get_global_id(0);
            if (i >= count) return;
            std::uint32_t const k = d_keys_in[i];
            std::uint64_t const v = d_vals_in[i];
            std::size_t const d = bucket_of_u32_dev(k, N, begin_bit, end_bit);
            sycl::atomic_ref<std::uint32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                cur(d_dst_cur[d]);
            std::uint32_t const slot = cur.fetch_add(1u);
            std::uint32_t const idx  = d_dst_offsets[d] + slot;
            d_staging_keys[idx] = k;
            d_staging_vals[idx] = v;
        }).wait();
}

void launch_scatter_u32_u64u32(
    std::uint32_t const* d_keys_in,
    std::uint64_t const* d_va_in, std::uint32_t const* d_vb_in,
    std::uint64_t count, std::uint32_t N, int begin_bit, int end_bit,
    std::uint32_t const* d_dst_offsets, std::uint32_t* d_dst_cur,
    std::uint32_t* d_staging_keys,
    std::uint64_t* d_staging_va, std::uint32_t* d_staging_vb,
    sycl::queue& q)
{
    if (count == 0) return;
    constexpr std::size_t threads = 256;
    std::size_t const groups = (count + threads - 1) / threads;
    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=](sycl::nd_item<1> it) {
            std::uint64_t i = it.get_global_id(0);
            if (i >= count) return;
            std::uint32_t const k  = d_keys_in[i];
            std::uint64_t const va = d_va_in[i];
            std::uint32_t const vb = d_vb_in[i];
            std::size_t const d = bucket_of_u32_dev(k, N, begin_bit, end_bit);
            sycl::atomic_ref<std::uint32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                cur(d_dst_cur[d]);
            std::uint32_t const slot = cur.fetch_add(1u);
            std::uint32_t const idx  = d_dst_offsets[d] + slot;
            d_staging_keys[idx] = k;
            d_staging_va[idx]   = va;
            d_staging_vb[idx]   = vb;
        }).wait();
}

void launch_scatter_u64_keys(
    std::uint64_t const* d_keys_in,
    std::uint64_t count, std::uint32_t N, int begin_bit, int end_bit,
    std::uint32_t const* d_dst_offsets, std::uint32_t* d_dst_cur,
    std::uint64_t* d_staging_keys,
    sycl::queue& q)
{
    if (count == 0) return;
    constexpr std::size_t threads = 256;
    std::size_t const groups = (count + threads - 1) / threads;
    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=](sycl::nd_item<1> it) {
            std::uint64_t i = it.get_global_id(0);
            if (i >= count) return;
            std::uint64_t const k = d_keys_in[i];
            std::size_t const d = bucket_of_u64_dev(k, N, begin_bit, end_bit);
            sycl::atomic_ref<std::uint32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                cur(d_dst_cur[d]);
            std::uint32_t const slot = cur.fetch_add(1u);
            std::uint32_t const idx  = d_dst_offsets[d] + slot;
            d_staging_keys[idx] = k;
        }).wait();
}

} // namespace pos2gpu::detail
