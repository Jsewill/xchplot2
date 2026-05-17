// StreamingPartition.cu — CUDA implementation of
// launch_streaming_partition_u32_u64 and the triple-val variant.
//
// Two-pass algorithm (direct translation of SYCL implementation):
//   Pass 1 (histogram): kernel walks d_keys_in once, atomic-incs
//     d_hist[bucket(key)]. D2H to host, exclusive-scan to get
//     h_bucket_starts. H2D the starts into d_cursors (atomic write
//     positions, initialized to bucket starts).
//   Pass 2 (per-tile partition):
//     For each tile of source positions [tile_off, tile_off + tile_n):
//       H2D h_vals_in[tile_off..] → d_vals_tile (device)
//       Partition kernel: for each i in tile, look up bucket from
//         d_keys_in[tile_off+i], atomicAdd d_cursors[bucket], write
//         (key, val) to (h_part_keys[slot], h_part_vals[slot]) via
//         the UVA-mapped host pinned pointers.
//
// UVA-mapped writes from kernels: cudaMallocHost-allocated host
// memory is device-accessible at the same virtual address. The
// writes go across PCIe — random, uncoalesced. That's the slow cost
// we accept in exchange for keeping the device peak low.

#include "gpu/StreamingPartition.cuh"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace pos2gpu {

namespace {

constexpr int kThreads = 256;

// Default tile size: ~16 MiB of u64 values per tile = 2M entries.
// Gives ~125 tiles at k=28 cap (~250M entries). Each H2D is well
// above PCIe latency-amortized threshold (~10 KB) so we're
// bandwidth-bound on the H2D, not latency-bound on dispatch.
constexpr uint64_t kDefaultTileEntries = uint64_t{1} << 21;  // 2M

inline uint64_t pick_tile_n(uint64_t count, uint64_t tile_count)
{
    if (tile_count == 0) {
        if (count <= kDefaultTileEntries) return 1;
        return (count + kDefaultTileEntries - 1) / kDefaultTileEntries;
    }
    return tile_count;
}

inline size_t align8(size_t b) { return (b + 7u) & ~size_t{7u}; }

__device__ inline uint32_t bucket_of(uint32_t key, int top_bit_offset, int num_top_bits)
{
    uint32_t const mask = (uint32_t{1} << num_top_bits) - 1u;
    return (key >> top_bit_offset) & mask;
}

__global__ void histogram_kernel(
    uint32_t const* __restrict__ d_keys_in,
    uint64_t count,
    int top_bit_offset,
    int num_top_bits,
    uint32_t* __restrict__ d_hist)
{
    uint64_t const i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t const b = bucket_of(d_keys_in[i], top_bit_offset, num_top_bits);
    atomicAdd(&d_hist[b], 1u);
}

__global__ void partition_kernel_u32_u64(
    uint32_t const* __restrict__ d_keys_tile,
    uint64_t const* __restrict__ d_vals_tile,
    uint64_t tile_n,
    int top_bit_offset,
    int num_top_bits,
    uint32_t* __restrict__ d_cursors,
    uint32_t* __restrict__ h_part_keys,
    uint64_t* __restrict__ h_part_vals)
{
    uint64_t const i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= tile_n) return;
    uint32_t const k = d_keys_tile[i];
    uint64_t const v = d_vals_tile[i];
    uint32_t const b = bucket_of(k, top_bit_offset, num_top_bits);
    uint32_t const pos = atomicAdd(&d_cursors[b], 1u);
    h_part_keys[pos] = k;
    h_part_vals[pos] = v;
}

__global__ void partition_kernel_u32_u64_u32(
    uint32_t const* __restrict__ d_keys_tile,
    uint64_t const* __restrict__ d_vals_tile,
    uint32_t const* __restrict__ d_vals2_tile,
    uint64_t tile_n,
    int top_bit_offset,
    int num_top_bits,
    uint32_t* __restrict__ d_cursors,
    uint32_t* __restrict__ h_part_keys,
    uint64_t* __restrict__ h_part_vals,
    uint32_t* __restrict__ h_part_vals2)
{
    uint64_t const i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= tile_n) return;
    uint32_t const k  = d_keys_tile[i];
    uint64_t const v  = d_vals_tile[i];
    uint32_t const v2 = d_vals2_tile[i];
    uint32_t const b  = bucket_of(k, top_bit_offset, num_top_bits);
    uint32_t const pos = atomicAdd(&d_cursors[b], 1u);
    h_part_keys[pos]  = k;
    h_part_vals[pos]  = v;
    h_part_vals2[pos] = v2;
}

} // namespace

cudaError_t launch_streaming_partition_u32_u64(
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
    cudaStream_t stream)
{
    if (num_top_bits < 1 || num_top_bits > 16) return cudaErrorInvalidValue;
    if (top_bit_offset < 0 || top_bit_offset + num_top_bits > 32) return cudaErrorInvalidValue;

    size_t const num_buckets = size_t{1} << num_top_bits;
    uint64_t const tiles     = pick_tile_n(count, tile_count);
    uint64_t const tile_size = (count + tiles - 1) / tiles;

    // d_scratch layout:
    //   [0, hist_aligned)        — d_hist / d_cursors (u32, num_buckets)
    //   [.. + vals_tile_bytes)   — d_vals_tile (u64, tile_size)
    size_t const hist_bytes      = num_buckets * sizeof(uint32_t);
    size_t const hist_aligned    = align8(hist_bytes);
    size_t const vals_tile_bytes = tile_size * sizeof(uint64_t);
    size_t const total_bytes     = hist_aligned + vals_tile_bytes;

    if (d_scratch == nullptr) {
        scratch_bytes = total_bytes;
        return cudaSuccess;
    }
    if (scratch_bytes < total_bytes) return cudaErrorInvalidValue;

    auto* base           = static_cast<unsigned char*>(d_scratch);
    auto* d_hist_cursors = reinterpret_cast<uint32_t*>(base);
    auto* d_vals_tile    = reinterpret_cast<uint64_t*>(base + hist_aligned);

    // Degenerate count: zero bucket starts via host (h_bucket_starts is
    // host-pinned but reachable from host as a normal pointer).
    for (size_t i = 0; i <= num_buckets; ++i) h_bucket_starts[i] = 0;
    if (count == 0) {
        scratch_bytes = total_bytes;
        return cudaSuccess;
    }

    cudaError_t err;

    // Pass 1: histogram.
    err = cudaMemsetAsync(d_hist_cursors, 0, hist_bytes, stream);
    if (err != cudaSuccess) return err;
    {
        uint64_t const blocks = (count + kThreads - 1) / kThreads;
        if (blocks > UINT_MAX) return cudaErrorInvalidValue;
        histogram_kernel<<<dim3(static_cast<unsigned>(blocks)), dim3(kThreads), 0, stream>>>(
            d_keys_in, count, top_bit_offset, num_top_bits, d_hist_cursors);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // D2H histogram → host scan → H2D cursors.
    std::vector<uint32_t> h_hist(num_buckets);
    err = cudaMemcpyAsync(h_hist.data(), d_hist_cursors,
                          num_buckets * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    uint32_t cum = 0;
    for (size_t b = 0; b < num_buckets; ++b) {
        h_bucket_starts[b] = cum;
        cum += h_hist[b];
    }
    h_bucket_starts[num_buckets] = cum;
    if (static_cast<uint64_t>(cum) != count) return cudaErrorUnknown;

    err = cudaMemcpyAsync(d_hist_cursors, h_bucket_starts,
                          num_buckets * sizeof(uint32_t),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;

    // Pass 2: per-tile partition with UVA-mapped writes to host pinned.
    for (uint64_t t = 0; t < tiles; ++t) {
        uint64_t const tile_off = t * tile_size;
        if (tile_off >= count) break;
        uint64_t const tile_n = std::min(tile_size, count - tile_off);

        err = cudaMemcpyAsync(d_vals_tile, h_vals_in + tile_off,
                              tile_n * sizeof(uint64_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) return err;

        uint64_t const blocks = (tile_n + kThreads - 1) / kThreads;
        if (blocks > UINT_MAX) return cudaErrorInvalidValue;
        partition_kernel_u32_u64<<<dim3(static_cast<unsigned>(blocks)),
                                   dim3(kThreads), 0, stream>>>(
            d_keys_in + tile_off, d_vals_tile, tile_n,
            top_bit_offset, num_top_bits,
            d_hist_cursors, h_part_keys, h_part_vals);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;

        // Per-tile sync: subsequent tile's H2D reuses d_vals_tile,
        // and the kernel's UVA writes to host need to land before the
        // caller observes the output.
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) return err;
    }

    scratch_bytes = total_bytes;
    return cudaSuccess;
}

cudaError_t launch_streaming_partition_u32_u64_u32(
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
    cudaStream_t stream)
{
    if (num_top_bits < 1 || num_top_bits > 16) return cudaErrorInvalidValue;
    if (top_bit_offset < 0 || top_bit_offset + num_top_bits > 32) return cudaErrorInvalidValue;

    size_t const num_buckets = size_t{1} << num_top_bits;
    uint64_t const tiles     = pick_tile_n(count, tile_count);
    uint64_t const tile_size = (count + tiles - 1) / tiles;

    // d_scratch layout:
    //   d_hist_cursors        (num_buckets × u32)
    //   d_vals_tile           (tile_size × u64)
    //   d_vals2_tile          (tile_size × u32)
    size_t const hist_bytes       = num_buckets * sizeof(uint32_t);
    size_t const hist_aligned     = align8(hist_bytes);
    size_t const vals_tile_bytes  = tile_size * sizeof(uint64_t);
    size_t const vals2_tile_bytes = tile_size * sizeof(uint32_t);
    size_t const vals2_aligned    = align8(vals2_tile_bytes);
    size_t const total_bytes      = hist_aligned + vals_tile_bytes + vals2_aligned;

    if (d_scratch == nullptr) {
        scratch_bytes = total_bytes;
        return cudaSuccess;
    }
    if (scratch_bytes < total_bytes) return cudaErrorInvalidValue;

    auto* base           = static_cast<unsigned char*>(d_scratch);
    auto* d_hist_cursors = reinterpret_cast<uint32_t*>(base);
    auto* d_vals_tile    = reinterpret_cast<uint64_t*>(base + hist_aligned);
    auto* d_vals2_tile   = reinterpret_cast<uint32_t*>(base + hist_aligned + vals_tile_bytes);

    for (size_t i = 0; i <= num_buckets; ++i) h_bucket_starts[i] = 0;
    if (count == 0) {
        scratch_bytes = total_bytes;
        return cudaSuccess;
    }

    cudaError_t err;

    // Pass 1: histogram (identical to u32_u64).
    err = cudaMemsetAsync(d_hist_cursors, 0, hist_bytes, stream);
    if (err != cudaSuccess) return err;
    {
        uint64_t const blocks = (count + kThreads - 1) / kThreads;
        if (blocks > UINT_MAX) return cudaErrorInvalidValue;
        histogram_kernel<<<dim3(static_cast<unsigned>(blocks)), dim3(kThreads), 0, stream>>>(
            d_keys_in, count, top_bit_offset, num_top_bits, d_hist_cursors);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    std::vector<uint32_t> h_hist(num_buckets);
    err = cudaMemcpyAsync(h_hist.data(), d_hist_cursors,
                          num_buckets * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    uint32_t cum = 0;
    for (size_t b = 0; b < num_buckets; ++b) {
        h_bucket_starts[b] = cum;
        cum += h_hist[b];
    }
    h_bucket_starts[num_buckets] = cum;
    if (static_cast<uint64_t>(cum) != count) return cudaErrorUnknown;

    err = cudaMemcpyAsync(d_hist_cursors, h_bucket_starts,
                          num_buckets * sizeof(uint32_t),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;

    // Pass 2: per-tile partition with two parallel val streams.
    for (uint64_t t = 0; t < tiles; ++t) {
        uint64_t const tile_off = t * tile_size;
        if (tile_off >= count) break;
        uint64_t const tile_n = std::min(tile_size, count - tile_off);

        err = cudaMemcpyAsync(d_vals_tile, h_vals_in + tile_off,
                              tile_n * sizeof(uint64_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) return err;
        err = cudaMemcpyAsync(d_vals2_tile, h_vals2_in + tile_off,
                              tile_n * sizeof(uint32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) return err;

        uint64_t const blocks = (tile_n + kThreads - 1) / kThreads;
        if (blocks > UINT_MAX) return cudaErrorInvalidValue;
        partition_kernel_u32_u64_u32<<<dim3(static_cast<unsigned>(blocks)),
                                       dim3(kThreads), 0, stream>>>(
            d_keys_in + tile_off, d_vals_tile, d_vals2_tile, tile_n,
            top_bit_offset, num_top_bits,
            d_hist_cursors, h_part_keys, h_part_vals, h_part_vals2);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) return err;
    }

    scratch_bytes = total_bytes;
    return cudaSuccess;
}

} // namespace pos2gpu
