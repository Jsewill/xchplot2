// sycl_streaming_partition_u32_u64_u32_parity — Phase 1.5a.
//
// Validates launch_streaming_partition_u32_u64_u32 by per-bucket
// multiset compare against a CPU reference. The triple-val
// variant is needed by Phase 1.5 (T2 sort streaming) where each
// entry carries a u64 meta AND a u32 xbits through the sort, and
// duplicate mi keys mean meta/xbits must travel together (running
// two separate u32_u64 partitions would unpair them across the
// non-deterministic atomic-claim ordering).
//
// Per-case invariants checked (same shape as
// sycl_streaming_partition_parity, extended for the triple val):
//   mono_ok       — h_bucket_starts monotone non-decreasing
//   total_ok      — h_bucket_starts[num_buckets] == count
//   counts_ok     — per-bucket count matches CPU reference
//   consistent_ok — every key in bucket b has top-bits == b
//   triples_ok    — per-bucket multiset of (key, val, val2) triples
//                   matches CPU reference. This is the load-bearing
//                   check: it asserts that meta+xbits stay PAIRED
//                   across the partition (i.e., output[i].val and
//                   output[i].val2 always came from the same input
//                   position).
//
// Test matrix: 3 seeds × 5 sizes (16..2^22) × 2 bucket counts
// (B=4, B=8). Duplicate keys are introduced deliberately (keys
// drawn with replacement) so the meta/xbits pairing invariant
// gets exercised on the case that motivated the primitive.

#include "gpu/StreamingPartition.cuh"
#include "gpu/SyclBackend.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <tuple>
#include <vector>

namespace {

bool run_case(uint32_t seed, uint64_t count, int num_top_bits,
              int top_bit_offset)
{
    auto& q = pos2gpu::sycl_backend::queue();
    size_t const num_buckets = size_t{1} << num_top_bits;
    uint32_t const mask = (uint32_t{1} << num_top_bits) - 1u;

    // Inputs. Keys drawn WITH REPLACEMENT to exercise the
    // duplicate-key pairing invariant. vals + vals2 are random.
    std::mt19937_64 rng(seed);
    uint32_t const key_range = (top_bit_offset + num_top_bits >= 32)
                                ? 0xFFFFFFFFu
                                : ((1u << (top_bit_offset + num_top_bits)) - 1u);
    std::vector<uint32_t> h_keys(count);
    std::vector<uint64_t> h_vals(count);
    std::vector<uint32_t> h_vals2(count);
    for (uint64_t i = 0; i < count; ++i) {
        h_keys[i]  = static_cast<uint32_t>(rng()) & key_range;
        h_vals[i]  = rng();
        h_vals2[i] = static_cast<uint32_t>(rng());
    }

    // Device-side keys; host-pinned vals + outputs.
    uint32_t* d_keys_in       = sycl::malloc_device<uint32_t>(count, q);
    uint64_t* h_pinned_vals   = sycl::malloc_host<uint64_t>(count, q);
    uint32_t* h_pinned_vals2  = sycl::malloc_host<uint32_t>(count, q);
    uint32_t* h_part_keys     = sycl::malloc_host<uint32_t>(count, q);
    uint64_t* h_part_vals     = sycl::malloc_host<uint64_t>(count, q);
    uint32_t* h_part_vals2    = sycl::malloc_host<uint32_t>(count, q);
    uint32_t* h_bucket_starts = sycl::malloc_host<uint32_t>(num_buckets + 1, q);

    q.memcpy(d_keys_in,      h_keys.data(),  sizeof(uint32_t) * count);
    q.memcpy(h_pinned_vals,  h_vals.data(),  sizeof(uint64_t) * count);
    q.memcpy(h_pinned_vals2, h_vals2.data(), sizeof(uint32_t) * count).wait();

    size_t scratch_bytes = 0;
    pos2gpu::launch_streaming_partition_u32_u64_u32(
        nullptr, scratch_bytes,
        d_keys_in, h_pinned_vals, h_pinned_vals2,
        h_part_keys, h_part_vals, h_part_vals2, h_bucket_starts,
        count, top_bit_offset, num_top_bits, /*tile_count=*/0, q);
    void* d_scratch = scratch_bytes ? sycl::malloc_device(scratch_bytes, q) : nullptr;

    auto const t0 = std::chrono::steady_clock::now();
    pos2gpu::launch_streaming_partition_u32_u64_u32(
        d_scratch, scratch_bytes,
        d_keys_in, h_pinned_vals, h_pinned_vals2,
        h_part_keys, h_part_vals, h_part_vals2, h_bucket_starts,
        count, top_bit_offset, num_top_bits, /*tile_count=*/0, q);
    q.wait();
    auto const t1 = std::chrono::steady_clock::now();
    double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // CPU reference: bucketize the same triples.
    std::vector<std::vector<std::tuple<uint32_t, uint64_t, uint32_t>>>
        ref_buckets(num_buckets);
    for (uint64_t i = 0; i < count; ++i) {
        uint32_t const b = (h_keys[i] >> top_bit_offset) & mask;
        ref_buckets[b].emplace_back(h_keys[i], h_vals[i], h_vals2[i]);
    }

    bool mono_ok = true;
    for (size_t b = 0; b < num_buckets; ++b) {
        if (h_bucket_starts[b] > h_bucket_starts[b + 1]) { mono_ok = false; break; }
    }
    bool const total_ok = (h_bucket_starts[num_buckets] == count);

    bool counts_ok = true;
    for (size_t b = 0; b < num_buckets; ++b) {
        if (h_bucket_starts[b + 1] - h_bucket_starts[b]
            != ref_buckets[b].size()) {
            counts_ok = false; break;
        }
    }

    bool consistent_ok = true;
    for (size_t b = 0; b < num_buckets && consistent_ok; ++b) {
        for (uint32_t p = h_bucket_starts[b]; p < h_bucket_starts[b + 1]; ++p) {
            if (((h_part_keys[p] >> top_bit_offset) & mask) != b) {
                consistent_ok = false; break;
            }
        }
    }

    // Triples multiset equality — the load-bearing check.
    bool triples_ok = true;
    for (size_t b = 0; b < num_buckets && triples_ok; ++b) {
        uint32_t const start = h_bucket_starts[b];
        uint32_t const stop  = h_bucket_starts[b + 1];
        std::vector<std::tuple<uint32_t, uint64_t, uint32_t>> got(stop - start);
        for (uint32_t i = 0; i + start < stop; ++i) {
            got[i] = std::make_tuple(h_part_keys[start + i],
                                     h_part_vals[start + i],
                                     h_part_vals2[start + i]);
        }
        auto ref = ref_buckets[b];
        std::sort(got.begin(), got.end());
        std::sort(ref.begin(), ref.end());
        if (got != ref) { triples_ok = false; break; }
    }

    if (d_scratch) sycl::free(d_scratch, q);
    sycl::free(d_keys_in, q);
    sycl::free(h_pinned_vals, q);
    sycl::free(h_pinned_vals2, q);
    sycl::free(h_part_keys, q);
    sycl::free(h_part_vals, q);
    sycl::free(h_part_vals2, q);
    sycl::free(h_bucket_starts, q);

    bool const ok = mono_ok && total_ok && counts_ok && consistent_ok && triples_ok;
    std::printf("%s  seed=%u n=%llu B=%d  [mono=%d total=%d counts=%d cons=%d triples=%d  %.2fms]\n",
                ok ? "PASS" : "FAIL", seed, (unsigned long long)count, num_top_bits,
                mono_ok, total_ok, counts_ok, consistent_ok, triples_ok, ms);
    return ok;
}

} // namespace

int main()
{
    auto& q = pos2gpu::sycl_backend::queue();
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    bool all_pass = true;
    for (uint32_t seed : { 1u, 7u, 31u }) {
        for (uint64_t n : { uint64_t{16},
                            uint64_t{1} << 14,
                            uint64_t{1} << 18,
                            uint64_t{1} << 20,
                            uint64_t{1} << 22 }) {
            for (int B : { 4, 8 }) {
                int const top_bit_offset = std::max(0, 22 - B);
                all_pass &= run_case(seed, n, B, top_bit_offset);
            }
        }
    }

    std::printf("\n==> %s\n", all_pass ? "ALL OK" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
