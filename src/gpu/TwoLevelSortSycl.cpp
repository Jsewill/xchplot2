// TwoLevelSortSycl.cpp — SYCL implementation of
// launch_two_level_sort_pairs_u32_u32. Pure SYCL; the per-bucket
// radix sort delegates to the backend-dispatched launch_sort_pairs
// in Sort.cuh so this file works equally on CUB-backed builds and
// pure-SYCL builds.

#include "gpu/TwoLevelSort.cuh"
#include "gpu/Sort.cuh"
#include "gpu/SyclBackend.hpp"

#include <algorithm>
#include <cstring>
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

// (end_bit - num_top_bits) is the shift; (num_top_bits) wide. The
// partition uses the high (num_top_bits) of the sort range to
// decide bucket; the per-bucket inner sort then handles
// [begin_bit, end_bit - num_top_bits).
inline uint32_t bucket_of(uint32_t key, int top_bit_offset, int num_top_bits)
{
    uint32_t const mask = (uint32_t{1} << num_top_bits) - 1u;
    return (key >> top_bit_offset) & mask;
}

// Histogram top-num_top_bits of d_keys into d_hist (num_buckets
// entries). Caller pre-zeroes d_hist.
void launch_histogram(
    uint32_t const* d_keys, uint32_t* d_hist,
    uint64_t count, int top_bit_offset, int num_top_bits,
    sycl::queue& q)
{
    q.parallel_for(
        sycl::nd_range<1>{ global_for(count), kThreads },
        [=](sycl::nd_item<1> it) {
            uint64_t const i = it.get_global_id(0);
            if (i >= count) return;
            uint32_t const b = bucket_of(d_keys[i], top_bit_offset, num_top_bits);
            sycl::atomic_ref<uint32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                slot(d_hist[b]);
            slot.fetch_add(1u);
        }).wait();
}

// Partition (d_keys_in, d_vals_in) into (d_keys_part, d_vals_part)
// using d_cursors as per-bucket atomic write positions. Caller
// pre-initialises d_cursors to the exclusive-scan offsets so each
// thread's fetch_add yields a unique slot in its bucket's range.
void launch_partition(
    uint32_t const* d_keys_in, uint32_t const* d_vals_in,
    uint32_t* d_keys_part, uint32_t* d_vals_part,
    uint32_t* d_cursors,
    uint64_t count, int top_bit_offset, int num_top_bits,
    sycl::queue& q)
{
    q.parallel_for(
        sycl::nd_range<1>{ global_for(count), kThreads },
        [=](sycl::nd_item<1> it) {
            uint64_t const i = it.get_global_id(0);
            if (i >= count) return;
            uint32_t const k = d_keys_in[i];
            uint32_t const v = d_vals_in[i];
            uint32_t const b = bucket_of(k, top_bit_offset, num_top_bits);
            sycl::atomic_ref<uint32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                cur(d_cursors[b]);
            uint32_t const pos = cur.fetch_add(1u);
            d_keys_part[pos] = k;
            d_vals_part[pos] = v;
        }).wait();
}

// Inner-scratch byte size: querying the segmented sort, which on
// the CUDA backend yields cub::DeviceSegmentedRadixSort's scratch
// (small — proportional to num_segments) and on other backends
// falls back to a per-segment scratch sized for the worst-case
// segment (= num_items).
size_t inner_sort_scratch_bytes(
    uint64_t count, int num_segments,
    int begin_bit, int end_bit_inner, sycl::queue& q)
{
    size_t s = 0;
    launch_segmented_sort_pairs_u32_u32(
        nullptr, s,
        static_cast<uint32_t const*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t const*>(nullptr), static_cast<uint32_t*>(nullptr),
        count, num_segments,
        static_cast<uint32_t const*>(nullptr), static_cast<uint32_t const*>(nullptr),
        begin_bit, end_bit_inner, q);
    return s;
}

// 8-byte align temp-storage slices so each sub-buffer's start is
// a respectable boundary for the device's loads.
inline size_t align8(size_t b) { return (b + 7u) & ~size_t{7u}; }

} // namespace

void launch_two_level_sort_pairs_u32_u32(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t* keys_in, uint32_t* keys_out,
    uint32_t* vals_in, uint32_t* vals_out,
    uint64_t count,
    int begin_bit, int end_bit,
    int num_top_bits,
    uint32_t* d_bucket_starts,
    sycl::queue& q)
{
    if (num_top_bits < 1 || num_top_bits > 16) {
        throw std::invalid_argument(
            "launch_two_level_sort_pairs_u32_u32: num_top_bits out of range "
            "(1..16)");
    }
    if (begin_bit + num_top_bits > end_bit) {
        throw std::invalid_argument(
            "launch_two_level_sort_pairs_u32_u32: begin_bit + num_top_bits "
            "must not exceed end_bit (need >=1 bit left for inner sort)");
    }

    int    const top_bit_offset = end_bit - num_top_bits;
    size_t const num_buckets    = size_t{1} << num_top_bits;

    // Layout of d_temp_storage:
    //   [0, inner_bytes)                                 — inner sort scratch
    //   [inner_bytes, inner_bytes + N*4)                 — d_keys_part
    //   [..,         + 2*N*4)                            — d_vals_part
    //   [..,         + 2*N*4 + num_buckets*4)            — d_cursors
    size_t const inner_bytes   = inner_sort_scratch_bytes(
        count, static_cast<int>(num_buckets), begin_bit, top_bit_offset, q);
    size_t const inner_aligned = align8(inner_bytes);
    size_t const part_bytes    = count * sizeof(uint32_t);
    size_t const cursor_bytes  = num_buckets * sizeof(uint32_t);
    size_t const total_bytes   = inner_aligned + 2 * part_bytes + cursor_bytes;

    if (d_temp_storage == nullptr) {
        temp_bytes = total_bytes;
        return;
    }
    if (temp_bytes < total_bytes) {
        throw std::invalid_argument(
            "launch_two_level_sort_pairs_u32_u32: temp_bytes too small");
    }

    auto* base = static_cast<unsigned char*>(d_temp_storage);
    void*     d_inner_scratch = base;
    auto*     d_keys_part     = reinterpret_cast<uint32_t*>(base + inner_aligned);
    auto*     d_vals_part     = reinterpret_cast<uint32_t*>(base + inner_aligned + part_bytes);
    auto*     d_cursors       = reinterpret_cast<uint32_t*>(base + inner_aligned + 2 * part_bytes);

    // Degenerate count: emit a zero-filled d_bucket_starts and skip
    // every actual pass. Avoids dispatching empty kernels — some
    // SYCL impls don't love nd_range with global_size == 0.
    q.memset(d_bucket_starts, 0, (num_buckets + 1) * sizeof(uint32_t)).wait();
    if (count == 0) {
        temp_bytes = total_bytes;
        return;
    }

    // ---- Pass 1: histogram by top bits.
    // We write counts into the front (num_buckets) slots of
    // d_bucket_starts to avoid an extra device alloc; the prefix
    // scan then turns counts into starts in-place, and we extend
    // by one (the [num_buckets] tail = total) at the end.
    q.memset(d_bucket_starts, 0, num_buckets * sizeof(uint32_t)).wait();
    launch_histogram(keys_in, d_bucket_starts, count, top_bit_offset, num_top_bits, q);

    // ---- Pass 2: exclusive scan of histogram on host.
    // num_buckets <= 65536 — a single D2H + H2D round-trip is
    // ~ 256 KB worst-case, well under 1 ms. Not worth a device
    // scan kernel.
    std::vector<uint32_t> h_hist(num_buckets);
    std::vector<uint32_t> h_starts(num_buckets + 1);
    q.memcpy(h_hist.data(), d_bucket_starts, num_buckets * sizeof(uint32_t)).wait();
    uint32_t cum = 0;
    for (size_t b = 0; b < num_buckets; ++b) {
        h_starts[b] = cum;
        cum += h_hist[b];
    }
    h_starts[num_buckets] = cum;
    if (static_cast<uint64_t>(cum) != count) {
        throw std::runtime_error(
            "launch_two_level_sort_pairs_u32_u32: histogram total mismatch");
    }
    q.memcpy(d_bucket_starts, h_starts.data(),
             (num_buckets + 1) * sizeof(uint32_t)).wait();
    q.memcpy(d_cursors, h_starts.data(), num_buckets * sizeof(uint32_t)).wait();

    // ---- Pass 3: partition (keys_in, vals_in) → (d_keys_part, d_vals_part).
    launch_partition(keys_in, vals_in, d_keys_part, d_vals_part,
                     d_cursors, count, top_bit_offset, num_top_bits, q);

    // ---- Pass 4: per-bucket sort.
    // Single segmented-sort call (one CUB launch on NVIDIA; a
    // per-segment loop on other backends — see SortSycl.cpp's
    // launch_segmented_sort_pairs_u32_u32_sycl). Result lands in
    // (keys_out, vals_out) at the partition's contiguous positions.
    //
    // Special case: if there are no inner bits to sort
    // (begin_bit == top_bit_offset), the partition output is
    // already in final order — just copy.
    if (begin_bit == top_bit_offset) {
        q.memcpy(keys_out, d_keys_part, count * sizeof(uint32_t));
        q.memcpy(vals_out, d_vals_part, count * sizeof(uint32_t)).wait();
    } else {
        size_t inner_bytes_actual = inner_bytes;
        // d_bucket_starts holds num_buckets+1 entries; segment i
        // occupies [d_bucket_starts[i], d_bucket_starts[i+1]).
        // Pass d_bucket_starts (begin) and d_bucket_starts + 1
        // (end) — both are valid for the first num_buckets entries.
        launch_segmented_sort_pairs_u32_u32(
            d_inner_scratch, inner_bytes_actual,
            d_keys_part, keys_out,
            d_vals_part, vals_out,
            count, static_cast<int>(num_buckets),
            d_bucket_starts, d_bucket_starts + 1,
            begin_bit, top_bit_offset, q);
    }
    q.wait();

    temp_bytes = total_bytes;
}

} // namespace pos2gpu
