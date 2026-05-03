// SortSycl.cpp — stable LSD radix sort in SYCL with parallel scan +
// per-tile parallel-across-tiles scatter. Used when XCHPLOT2_BUILD_CUDA=OFF;
// the CUDA build uses SortCuda.cu (CUB).
//
// Why hand-rolled? oneDPL's sort_by_key segfaults on AdaptiveCpp's CUDA
// backend, and AdaptiveCpp's bitonic_sort is O(N log² N) and unstable
// (we need stability for LSD radix). This implementation runs on every
// AdaptiveCpp backend (CUDA, HIP, Level Zero, OpenCL).
//
// Design (per 4-bit pass; RADIX=16; TILE_SIZE=1024 items per workgroup):
//   Phase 1 — parallel per-tile count: each WG reduces its tile into a
//     local 16-bucket histogram, then writes those 16 counts (no atomics)
//     into a bucket-major device array tile_hist[d * num_tiles + t]. The
//     bucket-major layout is what makes phase 2 a single 1-D scan.
//   Phase 2 — global exclusive scan over the entire tile_hist via
//     AdaptiveCpp's scanning::scan (decoupled-lookback, multi-WG, parallel).
//     The scan output, tile_offsets[d * num_tiles + t], is exactly the
//     starting position in the output where tile t's bucket-d items go,
//     because the bucket-major layout means the scan accumulates each
//     bucket's tiles in order, then rolls over to the next bucket. Stable
//     by construction: tile t < t' always lands earlier within bucket d.
//   Phase 3 — parallel-across-tiles scatter: each WG loads its tile into
//     local memory, then thread 0 sequentially walks the tile and emits
//     each item to out[tile_offsets[d * num_tiles + t] + pos[d]++]. Stable
//     within each tile (sequential walk preserves input order).
//
// Performance vs CUB: significantly slower (single-thread scatter per WG
// is ~32× under-utilized vs CUB's warp-cooperative scatter), but parallel
// across tiles. Future work: cooperative intra-tile scatter using per-WG
// per-bucket prefix scans. For now, correct and parallel beats fast and
// wrong.

#include "gpu/Sort.cuh"

#include <sycl/sycl.hpp>

#include "hipSYCL/algorithms/scan/scan.hpp"
#include "hipSYCL/algorithms/util/allocation_cache.hpp"

#include <cstdint>
#include <utility>

namespace pos2gpu {

namespace {

constexpr int  RADIX_BITS       = 4;
constexpr int  RADIX            = 1 << RADIX_BITS;
constexpr int  RADIX_MASK       = RADIX - 1;
constexpr int  WG_SIZE          = 256;
constexpr int  ITEMS_PER_THREAD = 4;
constexpr int  TILE_SIZE        = WG_SIZE * ITEMS_PER_THREAD;  // 1024

using local_atomic_u32 = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::work_group,
    sycl::access::address_space::local_space>;

// Per-process scratch cache for AdaptiveCpp's scan algorithm. Lives for
// the program's lifetime; allocations are pooled and reused across calls.
hipsycl::algorithms::util::allocation_cache& scan_alloc_cache()
{
    static hipsycl::algorithms::util::allocation_cache cache(
        hipsycl::algorithms::util::allocation_type::device);
    return cache;
}

uint64_t tile_count_for(uint64_t count)
{
    return (count + TILE_SIZE - 1) / TILE_SIZE;
}

void radix_pass_pairs_u32(
    sycl::queue& q,
    uint32_t const* in_keys, uint32_t const* in_vals,
    uint32_t* out_keys,      uint32_t* out_vals,
    uint32_t* tile_hist,     uint32_t* tile_offsets,
    uint64_t count, int bit)
{
    uint64_t const num_tiles = tile_count_for(count);
    uint64_t const grid      = num_tiles * WG_SIZE;

    // Phase 1: per-tile histogram → tile_hist[d * num_tiles + t].
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> local_hist(sycl::range<1>(RADIX), h);
        h.parallel_for(sycl::nd_range<1>(grid, WG_SIZE),
            [=](sycl::nd_item<1> it) {
                int const tid = static_cast<int>(it.get_local_id(0));
                uint64_t const tile = it.get_group(0);

                if (tid < RADIX) local_hist[tid] = 0;
                it.barrier(sycl::access::fence_space::local_space);

                uint64_t const base = tile * TILE_SIZE;
                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    uint64_t const idx = base + static_cast<uint64_t>(i) * WG_SIZE + tid;
                    if (idx < count) {
                        uint32_t const d = (in_keys[idx] >> bit) & RADIX_MASK;
                        local_atomic_u32(local_hist[d]).fetch_add(1u);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                if (tid < RADIX) {
                    tile_hist[static_cast<uint64_t>(tid) * num_tiles + tile] = local_hist[tid];
                }
            });
    });
    q.wait();

    // Phase 2: parallel exclusive scan over the entire tile_hist.
    {
        hipsycl::algorithms::util::allocation_group scratch_alloc(
            &scan_alloc_cache(), q.get_device());
        size_t const scan_size = static_cast<size_t>(RADIX) * static_cast<size_t>(num_tiles);
        hipsycl::algorithms::scanning::scan</*IsInclusive=*/false>(
            q, scratch_alloc,
            tile_hist, tile_hist + scan_size,
            tile_offsets,
            sycl::plus<uint32_t>{},
            uint32_t{0}).wait();
    }

    // Phase 3: per-tile stable scatter, cooperative across the WG.
    // Items are laid out in local memory CONTIGUOUSLY-PER-THREAD so that
    // the per-digit prefix scan (one per bucket; 16 iterations) yields
    // ranks in input order, preserving stability. Each iteration:
    //   1. Each thread counts its items that match the current digit.
    //   2. exclusive_scan_over_group turns those counts into per-thread
    //      offsets within the bucket.
    //   3. Each thread scatters its matching items to local_bases[d] +
    //      offset, advancing one position per matching item.
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> local_keys  (sycl::range<1>(TILE_SIZE), h);
        sycl::local_accessor<uint32_t, 1> local_vals  (sycl::range<1>(TILE_SIZE), h);
        sycl::local_accessor<uint8_t,  1> local_digits(sycl::range<1>(TILE_SIZE), h);
        sycl::local_accessor<uint32_t, 1> local_bases (sycl::range<1>(RADIX),     h);
        h.parallel_for(sycl::nd_range<1>(grid, WG_SIZE),
            [=](sycl::nd_item<1> it) {
                int const tid = static_cast<int>(it.get_local_id(0));
                uint64_t const tile = it.get_group(0);
                auto const grp = it.get_group();

                uint64_t const base = tile * TILE_SIZE;
                int const items_in_tile = static_cast<int>(
                    sycl::min<uint64_t>(TILE_SIZE, count - base));

                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    int const local_pos = tid * ITEMS_PER_THREAD + i;
                    if (local_pos < items_in_tile) {
                        uint32_t const k = in_keys[base + local_pos];
                        local_keys  [local_pos] = k;
                        local_vals  [local_pos] = in_vals[base + local_pos];
                        local_digits[local_pos] = static_cast<uint8_t>((k >> bit) & RADIX_MASK);
                    }
                }

                if (tid < RADIX) {
                    local_bases[tid] = tile_offsets[
                        static_cast<uint64_t>(tid) * num_tiles + tile];
                }
                it.barrier(sycl::access::fence_space::local_space);

                for (int d = 0; d < RADIX; ++d) {
                    uint32_t my_count = 0;
                    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                        int const local_pos = tid * ITEMS_PER_THREAD + i;
                        if (local_pos < items_in_tile && local_digits[local_pos] == d) {
                            ++my_count;
                        }
                    }

                    uint32_t const my_offset = sycl::exclusive_scan_over_group(
                        grp, my_count, sycl::plus<uint32_t>());

                    uint32_t pos_in_bucket = my_offset;
                    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                        int const local_pos = tid * ITEMS_PER_THREAD + i;
                        if (local_pos < items_in_tile && local_digits[local_pos] == d) {
                            uint32_t const target = local_bases[d] + pos_in_bucket;
                            out_keys[target] = local_keys[local_pos];
                            out_vals[target] = local_vals[local_pos];
                            ++pos_in_bucket;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
            });
    });
    q.wait();
}

void radix_pass_keys_u64(
    sycl::queue& q,
    uint64_t const* in_keys,
    uint64_t* out_keys,
    uint32_t* tile_hist, uint32_t* tile_offsets,
    uint64_t count, int bit)
{
    uint64_t const num_tiles = tile_count_for(count);
    uint64_t const grid      = num_tiles * WG_SIZE;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> local_hist(sycl::range<1>(RADIX), h);
        h.parallel_for(sycl::nd_range<1>(grid, WG_SIZE),
            [=](sycl::nd_item<1> it) {
                int const tid = static_cast<int>(it.get_local_id(0));
                uint64_t const tile = it.get_group(0);

                if (tid < RADIX) local_hist[tid] = 0;
                it.barrier(sycl::access::fence_space::local_space);

                uint64_t const base = tile * TILE_SIZE;
                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    uint64_t const idx = base + static_cast<uint64_t>(i) * WG_SIZE + tid;
                    if (idx < count) {
                        uint32_t const d =
                            static_cast<uint32_t>((in_keys[idx] >> bit) & uint64_t{RADIX_MASK});
                        local_atomic_u32(local_hist[d]).fetch_add(1u);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                if (tid < RADIX) {
                    tile_hist[static_cast<uint64_t>(tid) * num_tiles + tile] = local_hist[tid];
                }
            });
    });
    q.wait();

    {
        hipsycl::algorithms::util::allocation_group scratch_alloc(
            &scan_alloc_cache(), q.get_device());
        size_t const scan_size = static_cast<size_t>(RADIX) * static_cast<size_t>(num_tiles);
        hipsycl::algorithms::scanning::scan</*IsInclusive=*/false>(
            q, scratch_alloc,
            tile_hist, tile_hist + scan_size,
            tile_offsets,
            sycl::plus<uint32_t>{},
            uint32_t{0}).wait();
    }

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint64_t, 1> local_keys  (sycl::range<1>(TILE_SIZE), h);
        sycl::local_accessor<uint8_t,  1> local_digits(sycl::range<1>(TILE_SIZE), h);
        sycl::local_accessor<uint32_t, 1> local_bases (sycl::range<1>(RADIX),     h);
        h.parallel_for(sycl::nd_range<1>(grid, WG_SIZE),
            [=](sycl::nd_item<1> it) {
                int const tid = static_cast<int>(it.get_local_id(0));
                uint64_t const tile = it.get_group(0);
                auto const grp = it.get_group();

                uint64_t const base = tile * TILE_SIZE;
                int const items_in_tile = static_cast<int>(
                    sycl::min<uint64_t>(TILE_SIZE, count - base));

                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    int const local_pos = tid * ITEMS_PER_THREAD + i;
                    if (local_pos < items_in_tile) {
                        uint64_t const k = in_keys[base + local_pos];
                        local_keys  [local_pos] = k;
                        local_digits[local_pos] =
                            static_cast<uint8_t>((k >> bit) & uint64_t{RADIX_MASK});
                    }
                }

                if (tid < RADIX) {
                    local_bases[tid] = tile_offsets[
                        static_cast<uint64_t>(tid) * num_tiles + tile];
                }
                it.barrier(sycl::access::fence_space::local_space);

                for (int d = 0; d < RADIX; ++d) {
                    uint32_t my_count = 0;
                    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                        int const local_pos = tid * ITEMS_PER_THREAD + i;
                        if (local_pos < items_in_tile && local_digits[local_pos] == d) {
                            ++my_count;
                        }
                    }

                    uint32_t const my_offset = sycl::exclusive_scan_over_group(
                        grp, my_count, sycl::plus<uint32_t>());

                    uint32_t pos_in_bucket = my_offset;
                    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                        int const local_pos = tid * ITEMS_PER_THREAD + i;
                        if (local_pos < items_in_tile && local_digits[local_pos] == d) {
                            uint32_t const target = local_bases[d] + pos_in_bucket;
                            out_keys[target] = local_keys[local_pos];
                            ++pos_in_bucket;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
            });
    });
    q.wait();
}

} // namespace

// DoubleBuffer-style ping-pong over caller's buffers — no internal alt
// allocation. Scratch is just tile_hist + tile_offsets (a few MB at k=28
// vs the ~6 GB the old keys_alt/vals_alt cost there). The result lands
// in keys_out; if the pass count is odd we do one final memcpy from
// keys_in (which holds the result after the last swap).
// Renamed _sycl in 2026-05; the canonical launch_sort_pairs_u32_u32 lives
// in SortDispatch.cpp and routes to this implementation for non-CUDA
// devices (and for everything when XCHPLOT2_HAVE_CUB isn't defined).
void launch_sort_pairs_u32_u32_sycl(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t* keys_in, uint32_t* keys_out,
    uint32_t* vals_in, uint32_t* vals_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q)
{
    uint64_t const num_tiles = tile_count_for(count);
    size_t const bytes = sizeof(uint32_t) * RADIX * num_tiles * 2;
    if (d_temp_storage == nullptr) {
        temp_bytes = bytes;
        return;
    }

    uint8_t* p = static_cast<uint8_t*>(d_temp_storage);
    uint32_t* tile_hist    = reinterpret_cast<uint32_t*>(p);  p += sizeof(uint32_t) * RADIX * num_tiles;
    uint32_t* tile_offsets = reinterpret_cast<uint32_t*>(p);

    // First pass reads from keys_in (caller's input). Subsequent passes
    // ping-pong between keys_in and keys_out — we treat keys_in as
    // scratch from here on, which the public API documents.
    uint32_t* cur_keys = keys_in;
    uint32_t* cur_vals = vals_in;
    uint32_t* dst_keys = keys_out;
    uint32_t* dst_vals = vals_out;

    for (int bit = begin_bit; bit < end_bit; bit += RADIX_BITS) {
        radix_pass_pairs_u32(q, cur_keys, cur_vals, dst_keys, dst_vals,
                             tile_hist, tile_offsets, count, bit);
        std::swap(cur_keys, dst_keys);
        std::swap(cur_vals, dst_vals);
    }
    q.wait();

    // After the loop, cur_keys/cur_vals point to the buffer holding the
    // sorted result (because radix_pass writes to dst, then we swap so
    // dst becomes the input for the next pass). If that's not keys_out,
    // copy the result over.
    if (cur_keys != keys_out) {
        q.memcpy(keys_out, cur_keys, sizeof(uint32_t) * count);
        q.memcpy(vals_out, cur_vals, sizeof(uint32_t) * count).wait();
    }
}

void launch_sort_keys_u64_sycl(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint64_t* keys_in, uint64_t* keys_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q)
{
    uint64_t const num_tiles = tile_count_for(count);
    size_t const bytes = sizeof(uint32_t) * RADIX * num_tiles * 2;
    if (d_temp_storage == nullptr) {
        temp_bytes = bytes;
        return;
    }

    uint8_t* p = static_cast<uint8_t*>(d_temp_storage);
    uint32_t* tile_hist    = reinterpret_cast<uint32_t*>(p);  p += sizeof(uint32_t) * RADIX * num_tiles;
    uint32_t* tile_offsets = reinterpret_cast<uint32_t*>(p);

    uint64_t* cur = keys_in;
    uint64_t* dst = keys_out;

    for (int bit = begin_bit; bit < end_bit; bit += RADIX_BITS) {
        radix_pass_keys_u64(q, cur, dst, tile_hist, tile_offsets, count, bit);
        std::swap(cur, dst);
    }
    q.wait();

    if (cur != keys_out) {
        q.memcpy(keys_out, cur, sizeof(uint64_t) * count).wait();
    }
}

} // namespace pos2gpu
