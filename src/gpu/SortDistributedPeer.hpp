// SortDistributedPeer.hpp — internal helpers shared by SortDistributed.cpp
// (entry points) and SortDistributedPeer.cpp (peer-path scatter kernels).
//
// Phase 2.4b's GPU-scatter peer transport needs:
//   1. A device-side bucket-of mapping that mirrors the host bucket_of_*
//      formula bit-for-bit so a key routed to dest d on host produces the
//      same d on device.
//   2. Two-pass count + scatter kernels per (key type, val tuple) variant.
//   3. A host-side prefix-scan helper.
//
// Defined here once; both TUs include this header. Not part of the public
// SortDistributed API — these are implementation details.

#pragma once

#include <cstdint>
#include <cstddef>

#include <sycl/sycl.hpp>

namespace pos2gpu::detail {

// ---------------------------------------------------------------------
// Host-side prefix-scan over per-destination counts. counts[N] →
// offsets[N], returns total. Trivial; not worth a kernel for N ≤ 32.
// ---------------------------------------------------------------------
inline std::uint32_t exclusive_scan_u32(
    std::uint32_t const* counts, std::uint32_t* offsets, std::size_t N)
{
    std::uint32_t cum = 0;
    for (std::size_t i = 0; i < N; ++i) {
        offsets[i] = cum;
        cum += counts[i];
    }
    return cum;
}

// ---------------------------------------------------------------------
// Pass-1 count kernels — one per key type. d_dst_count is N entries on
// the same device as the queue; caller pre-zeroes via memset.
// ---------------------------------------------------------------------
void launch_count_per_dest_u32(
    std::uint32_t const* d_keys, std::uint64_t count,
    std::uint32_t N, int begin_bit, int end_bit,
    std::uint32_t* d_dst_count, sycl::queue& q);

void launch_count_per_dest_u64(
    std::uint64_t const* d_keys, std::uint64_t count,
    std::uint32_t N, int begin_bit, int end_bit,
    std::uint32_t* d_dst_count, sycl::queue& q);

// ---------------------------------------------------------------------
// Pass-2 scatter kernels — one per (key type, val tuple) shape. Atomic
// slot allocation via d_dst_cur[N] (caller pre-zeroes). Same ordering
// non-determinism caveat as the entry points: tie order within a
// destination is arbitrary but the multiset is preserved.
// ---------------------------------------------------------------------
void launch_scatter_u32_u32(
    std::uint32_t const* d_keys_in, std::uint32_t const* d_vals_in,
    std::uint64_t count, std::uint32_t N, int begin_bit, int end_bit,
    std::uint32_t const* d_dst_offsets, std::uint32_t* d_dst_cur,
    std::uint32_t* d_staging_keys, std::uint32_t* d_staging_vals,
    sycl::queue& q);

void launch_scatter_u32_u64(
    std::uint32_t const* d_keys_in, std::uint64_t const* d_vals_in,
    std::uint64_t count, std::uint32_t N, int begin_bit, int end_bit,
    std::uint32_t const* d_dst_offsets, std::uint32_t* d_dst_cur,
    std::uint32_t* d_staging_keys, std::uint64_t* d_staging_vals,
    sycl::queue& q);

void launch_scatter_u32_u64u32(
    std::uint32_t const* d_keys_in,
    std::uint64_t const* d_va_in, std::uint32_t const* d_vb_in,
    std::uint64_t count, std::uint32_t N, int begin_bit, int end_bit,
    std::uint32_t const* d_dst_offsets, std::uint32_t* d_dst_cur,
    std::uint32_t* d_staging_keys,
    std::uint64_t* d_staging_va, std::uint32_t* d_staging_vb,
    sycl::queue& q);

void launch_scatter_u64_keys(
    std::uint64_t const* d_keys_in,
    std::uint64_t count, std::uint32_t N, int begin_bit, int end_bit,
    std::uint32_t const* d_dst_offsets, std::uint32_t* d_dst_cur,
    std::uint64_t* d_staging_keys,
    sycl::queue& q);

} // namespace pos2gpu::detail
