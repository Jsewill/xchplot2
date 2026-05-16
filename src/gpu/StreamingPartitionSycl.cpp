// StreamingPartitionSycl.cpp — SYCL implementation of
// launch_streaming_partition_u32_u64.
//
// Two-pass algorithm:
//   Pass 1 (histogram): kernel walks d_keys_in once, atomic-incs
//     d_hist[bucket(key)]. D2H to host, exclusive-scan to get
//     h_bucket_starts. H2D the starts into d_cursors (atomic
//     write positions, initialized to bucket starts).
//   Pass 2 (per-tile partition):
//     For each tile of source positions [tile_off, tile_off + tile_n):
//       H2D h_vals_in[tile_off..] → d_vals_tile (device)
//       Partition kernel: for each i in tile, look up bucket from
//         d_keys_in[tile_off+i], atomic-fetch-add d_cursors[bucket],
//         write (key, val) to (h_part_keys[slot], h_part_vals[slot])
//         via the USM-host pointers (zero-copy from kernel).
//
// USM-host writes from kernels: malloc_host allocations are
// device-accessible via USM. The writes go across PCIe — random,
// uncoalesced. That's the slow cost we accept in exchange for
// keeping the device peak low.

#include "gpu/StreamingPartition.cuh"
#include "gpu/SyclBackend.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

namespace pos2gpu {

namespace {

constexpr size_t kThreads = 256;

inline size_t global_for(uint64_t count)
{
    size_t const groups = static_cast<size_t>((count + kThreads - 1) / kThreads);
    return groups * kThreads;
}

inline uint32_t bucket_of(uint32_t key, int top_bit_offset, int num_top_bits)
{
    uint32_t const mask = (uint32_t{1} << num_top_bits) - 1u;
    return (key >> top_bit_offset) & mask;
}

// Default tile size: ~16 MiB of u64 values per tile = 2M entries.
// Gives ~125 tiles at k=28 cap (~250M entries). Each H2D is well
// above PCIe latency-amortized threshold (~10 KB) so we're
// bandwidth-bound on the H2D, not latency-bound on dispatch.
constexpr uint64_t kDefaultTileEntries = uint64_t{1} << 21;  // 2M

inline uint64_t pick_tile_n(uint64_t count, uint64_t tile_count)
{
    if (tile_count == 0) {
        // Caller didn't pick. Default: ~2M entries per tile, but at
        // least 1 tile. Returns the NUMBER OF TILES, not the entries
        // per tile (the kernel below divides count by this to get
        // per-tile entry count).
        if (count <= kDefaultTileEntries) return 1;
        return (count + kDefaultTileEntries - 1) / kDefaultTileEntries;
    }
    return tile_count;
}

inline size_t align8(size_t b) { return (b + 7u) & ~size_t{7u}; }

} // namespace

void launch_streaming_partition_u32_u64(
    void* d_scratch,
    size_t& scratch_bytes,
    uint32_t const* d_keys_in,
    uint64_t const* h_vals_in,
    uint32_t* h_part_keys,
    uint64_t* h_part_vals,
    uint32_t* h_bucket_starts,
    uint64_t count,
    int top_bit_offset,
    int num_top_bits,
    uint64_t tile_count,
    sycl::queue& q)
{
    if (num_top_bits < 1 || num_top_bits > 16) {
        throw std::invalid_argument(
            "launch_streaming_partition_u32_u64: num_top_bits out of range");
    }
    if (top_bit_offset < 0 || top_bit_offset + num_top_bits > 32) {
        throw std::invalid_argument(
            "launch_streaming_partition_u32_u64: top_bit_offset + num_top_bits out of range");
    }

    size_t const num_buckets   = size_t{1} << num_top_bits;
    uint64_t const tiles       = pick_tile_n(count, tile_count);
    uint64_t const tile_size   = (count + tiles - 1) / tiles;

    // Layout of d_scratch:
    //   [0, hist_bytes)         — d_hist / d_cursors (u32)
    //                             We use one buffer that doubles as
    //                             histogram (pass 1) then cursors (pass 2).
    //   [.. + vals_tile_bytes)  — d_vals_tile (u64) for the per-tile H2D
    size_t const hist_bytes        = num_buckets * sizeof(uint32_t);
    size_t const hist_aligned      = align8(hist_bytes);
    size_t const vals_tile_bytes   = tile_size * sizeof(uint64_t);
    size_t const total_bytes       = hist_aligned + vals_tile_bytes;

    if (d_scratch == nullptr) {
        scratch_bytes = total_bytes;
        return;
    }
    if (scratch_bytes < total_bytes) {
        throw std::invalid_argument(
            "launch_streaming_partition_u32_u64: scratch_bytes too small");
    }

    auto* base = static_cast<unsigned char*>(d_scratch);
    auto* d_hist_cursors = reinterpret_cast<uint32_t*>(base);
    auto* d_vals_tile    = reinterpret_cast<uint64_t*>(base + hist_aligned);

    // Degenerate count: zero the bucket starts and return.
    q.memset(h_bucket_starts, 0, (num_buckets + 1) * sizeof(uint32_t)).wait();
    if (count == 0) return;

    // ---- Pass 1: histogram over d_keys_in.
    q.memset(d_hist_cursors, 0, hist_bytes).wait();
    q.parallel_for(
        sycl::nd_range<1>{ global_for(count), kThreads },
        [=](sycl::nd_item<1> it) {
            uint64_t const i = it.get_global_id(0);
            if (i >= count) return;
            uint32_t const b = bucket_of(d_keys_in[i], top_bit_offset, num_top_bits);
            sycl::atomic_ref<uint32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                slot(d_hist_cursors[b]);
            slot.fetch_add(1u);
        }).wait();

    // ---- Pass 2 setup: D2H counts, scan, write back as cursors.
    std::vector<uint32_t> h_hist(num_buckets);
    q.memcpy(h_hist.data(), d_hist_cursors, num_buckets * sizeof(uint32_t)).wait();
    uint32_t cum = 0;
    for (size_t b = 0; b < num_buckets; ++b) {
        h_bucket_starts[b] = cum;
        cum += h_hist[b];
    }
    h_bucket_starts[num_buckets] = cum;
    if (static_cast<uint64_t>(cum) != count) {
        throw std::runtime_error(
            "launch_streaming_partition_u32_u64: histogram total mismatch");
    }
    // Cursors start at bucket starts; atomic-fetch-add yields the
    // next write slot for that bucket.
    q.memcpy(d_hist_cursors, h_bucket_starts, num_buckets * sizeof(uint32_t)).wait();

    // ---- Pass 3: per-tile partition with zero-copy writes to host.
    // d_keys_in stays full-cap on device throughout; we slice into
    // it with d_keys_in + tile_off. d_vals_tile is the only sizeable
    // working buffer.
    //
    // Per-iteration steps must serialize (the partition kernel
    // depends on the H2D having landed before it reads d_vals_tile).
    // We don't pipeline tile N's H2D with tile N-1's kernel here —
    // that's a Phase 1.3c optimization to attempt if measured
    // throughput on real target hardware demands it. First cut:
    // straight serial.
    for (uint64_t t = 0; t < tiles; ++t) {
        uint64_t const tile_off = t * tile_size;
        if (tile_off >= count) break;
        uint64_t const tile_n   = std::min(tile_size, count - tile_off);

        q.memcpy(d_vals_tile, h_vals_in + tile_off,
                 tile_n * sizeof(uint64_t)).wait();

        // Capture-by-value: the kernel writes through h_part_keys /
        // h_part_vals which are USM-host pointers. SYCL guarantees
        // these are device-accessible. AdaptiveCpp maps them through
        // the CUDA backend's mapped-host mechanism (or equivalent on
        // HIP/L0/CPU).
        uint32_t* const part_keys = h_part_keys;
        uint64_t* const part_vals = h_part_vals;
        uint32_t const* const keys_tile = d_keys_in + tile_off;
        q.parallel_for(
            sycl::nd_range<1>{ global_for(tile_n), kThreads },
            [=](sycl::nd_item<1> it) {
                uint64_t const i = it.get_global_id(0);
                if (i >= tile_n) return;
                uint32_t const k = keys_tile[i];
                uint64_t const v = d_vals_tile[i];
                uint32_t const b = bucket_of(k, top_bit_offset, num_top_bits);
                sycl::atomic_ref<uint32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
                    cur(d_hist_cursors[b]);
                uint32_t const pos = cur.fetch_add(1u);
                part_keys[pos] = k;
                part_vals[pos] = v;
            }).wait();
    }

    scratch_bytes = total_bytes;
}

// Triple-val variant. See StreamingPartition.cuh for the why.
// Layout-wise this is u32_u64 with an extra (d_vals2_tile, vals2)
// pair carried alongside. Both per-tile H2Ds happen back-to-back
// (single wait at end), and the partition kernel writes both
// outputs at the same atomic-claim slot.
void launch_streaming_partition_u32_u64_u32(
    void* d_scratch,
    size_t& scratch_bytes,
    uint32_t const* d_keys_in,
    uint64_t const* h_vals_in,
    uint32_t const* h_vals2_in,
    uint32_t* h_part_keys,
    uint64_t* h_part_vals,
    uint32_t* h_part_vals2,
    uint32_t* h_bucket_starts,
    uint64_t count,
    int top_bit_offset,
    int num_top_bits,
    uint64_t tile_count,
    sycl::queue& q)
{
    if (num_top_bits < 1 || num_top_bits > 16) {
        throw std::invalid_argument(
            "launch_streaming_partition_u32_u64_u32: num_top_bits out of range");
    }
    if (top_bit_offset < 0 || top_bit_offset + num_top_bits > 32) {
        throw std::invalid_argument(
            "launch_streaming_partition_u32_u64_u32: top_bit_offset + num_top_bits out of range");
    }

    size_t const num_buckets   = size_t{1} << num_top_bits;
    uint64_t const tiles       = pick_tile_n(count, tile_count);
    uint64_t const tile_size   = (count + tiles - 1) / tiles;

    // Layout of d_scratch:
    //   d_hist_cursors        (num_buckets × u32)
    //   d_vals_tile           (tile_size × u64)
    //   d_vals2_tile          (tile_size × u32)
    size_t const hist_bytes        = num_buckets * sizeof(uint32_t);
    size_t const hist_aligned      = align8(hist_bytes);
    size_t const vals_tile_bytes   = tile_size * sizeof(uint64_t);
    size_t const vals2_tile_bytes  = tile_size * sizeof(uint32_t);
    size_t const vals2_aligned     = align8(vals2_tile_bytes);
    size_t const total_bytes       = hist_aligned + vals_tile_bytes + vals2_aligned;

    if (d_scratch == nullptr) {
        scratch_bytes = total_bytes;
        return;
    }
    if (scratch_bytes < total_bytes) {
        throw std::invalid_argument(
            "launch_streaming_partition_u32_u64_u32: scratch_bytes too small");
    }

    auto* base = static_cast<unsigned char*>(d_scratch);
    auto* d_hist_cursors = reinterpret_cast<uint32_t*>(base);
    auto* d_vals_tile    = reinterpret_cast<uint64_t*>(base + hist_aligned);
    auto* d_vals2_tile   = reinterpret_cast<uint32_t*>(base + hist_aligned + vals_tile_bytes);

    q.memset(h_bucket_starts, 0, (num_buckets + 1) * sizeof(uint32_t)).wait();
    if (count == 0) return;

    // Pass 1: histogram (identical to u32_u64).
    q.memset(d_hist_cursors, 0, hist_bytes).wait();
    q.parallel_for(
        sycl::nd_range<1>{ global_for(count), kThreads },
        [=](sycl::nd_item<1> it) {
            uint64_t const i = it.get_global_id(0);
            if (i >= count) return;
            uint32_t const b = bucket_of(d_keys_in[i], top_bit_offset, num_top_bits);
            sycl::atomic_ref<uint32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                slot(d_hist_cursors[b]);
            slot.fetch_add(1u);
        }).wait();

    std::vector<uint32_t> h_hist(num_buckets);
    q.memcpy(h_hist.data(), d_hist_cursors, num_buckets * sizeof(uint32_t)).wait();
    uint32_t cum = 0;
    for (size_t b = 0; b < num_buckets; ++b) {
        h_bucket_starts[b] = cum;
        cum += h_hist[b];
    }
    h_bucket_starts[num_buckets] = cum;
    if (static_cast<uint64_t>(cum) != count) {
        throw std::runtime_error(
            "launch_streaming_partition_u32_u64_u32: histogram total mismatch");
    }
    q.memcpy(d_hist_cursors, h_bucket_starts, num_buckets * sizeof(uint32_t)).wait();

    // Pass 2: per-tile partition with two parallel val streams.
    for (uint64_t t = 0; t < tiles; ++t) {
        uint64_t const tile_off = t * tile_size;
        if (tile_off >= count) break;
        uint64_t const tile_n   = std::min(tile_size, count - tile_off);

        q.memcpy(d_vals_tile,  h_vals_in  + tile_off,
                 tile_n * sizeof(uint64_t));
        q.memcpy(d_vals2_tile, h_vals2_in + tile_off,
                 tile_n * sizeof(uint32_t)).wait();

        uint32_t* const part_keys  = h_part_keys;
        uint64_t* const part_vals  = h_part_vals;
        uint32_t* const part_vals2 = h_part_vals2;
        uint32_t const* const keys_tile = d_keys_in + tile_off;
        q.parallel_for(
            sycl::nd_range<1>{ global_for(tile_n), kThreads },
            [=](sycl::nd_item<1> it) {
                uint64_t const i = it.get_global_id(0);
                if (i >= tile_n) return;
                uint32_t const k  = keys_tile[i];
                uint64_t const v  = d_vals_tile[i];
                uint32_t const v2 = d_vals2_tile[i];
                uint32_t const b  = bucket_of(k, top_bit_offset, num_top_bits);
                sycl::atomic_ref<uint32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
                    cur(d_hist_cursors[b]);
                uint32_t const pos = cur.fetch_add(1u);
                part_keys[pos]  = k;
                part_vals[pos]  = v;
                part_vals2[pos] = v2;
            }).wait();
    }

    scratch_bytes = total_bytes;
}

} // namespace pos2gpu
