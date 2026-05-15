// sycl_streaming_partition_parity — Phase 1.3b. Validates
// launch_streaming_partition_u32_u64.
//
// The streaming partition writes to bucket arenas via atomic-claim,
// so the order within each bucket is unspecified. To verify
// correctness we compute, for each bucket, the SET of (key, val)
// pairs produced by the streaming primitive vs. a CPU reference
// that does the same bucketing. The two sets must be equal as
// multisets — same elements, same counts.
//
// Additional invariants checked:
//   - h_bucket_starts is monotone non-decreasing.
//   - h_bucket_starts[num_buckets] == count.
//   - Every key in bucket b has top-num_top_bits == b.
//
// Test matrix mirrors sycl_two_level_sort_parity: 3 seeds × 5 sizes
// (16..2^22) × 2 bucket counts (B=4, B=8). Vals are random u64;
// keys are unique shuffled u32 over [0, count) so the bucket
// assignments are well-distributed.

#include "gpu/StreamingPartition.cuh"
#include "gpu/SyclBackend.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

bool run_case(uint32_t seed, uint64_t count, int num_top_bits,
              int top_bit_offset)
{
    auto& q = pos2gpu::sycl_backend::queue();
    size_t const num_buckets = size_t{1} << num_top_bits;
    uint32_t const mask = (uint32_t{1} << num_top_bits) - 1u;

    // Build input: unique keys, random vals.
    std::mt19937_64 rng(seed);
    std::vector<uint32_t> h_keys(count);
    std::vector<uint64_t> h_vals(count);
    for (uint64_t i = 0; i < count; ++i) {
        h_keys[i] = static_cast<uint32_t>(i);
        h_vals[i] = rng();
    }
    std::shuffle(h_keys.begin(), h_keys.end(), rng);

    // Device-side keys; host-pinned vals + outputs.
    uint32_t* d_keys_in       = sycl::malloc_device<uint32_t>(count, q);
    uint64_t* h_pinned_vals   = sycl::malloc_host<uint64_t>(count, q);
    uint32_t* h_part_keys     = sycl::malloc_host<uint32_t>(count, q);
    uint64_t* h_part_vals     = sycl::malloc_host<uint64_t>(count, q);
    uint32_t* h_bucket_starts = sycl::malloc_host<uint32_t>(num_buckets + 1, q);

    q.memcpy(d_keys_in,     h_keys.data(), sizeof(uint32_t) * count);
    q.memcpy(h_pinned_vals, h_vals.data(), sizeof(uint64_t) * count).wait();

    // Query + allocate scratch.
    size_t scratch_bytes = 0;
    pos2gpu::launch_streaming_partition_u32_u64(
        nullptr, scratch_bytes,
        d_keys_in, h_pinned_vals,
        h_part_keys, h_part_vals, h_bucket_starts,
        count, top_bit_offset, num_top_bits,
        /*tile_count=*/0, q);
    void* d_scratch = scratch_bytes ? sycl::malloc_device(scratch_bytes, q) : nullptr;

    auto const t0 = std::chrono::steady_clock::now();
    pos2gpu::launch_streaming_partition_u32_u64(
        d_scratch, scratch_bytes,
        d_keys_in, h_pinned_vals,
        h_part_keys, h_part_vals, h_bucket_starts,
        count, top_bit_offset, num_top_bits,
        /*tile_count=*/0, q);
    q.wait();
    auto const t1 = std::chrono::steady_clock::now();
    double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // CPU reference: bucketize.
    std::vector<std::vector<uint64_t>> ref_buckets_keys(num_buckets);
    std::vector<std::vector<uint64_t>> ref_buckets_vals(num_buckets);
    for (uint64_t i = 0; i < count; ++i) {
        uint32_t const b = (h_keys[i] >> top_bit_offset) & mask;
        ref_buckets_keys[b].push_back(h_keys[i]);
        ref_buckets_vals[b].push_back(h_vals[i]);
    }

    // Validation 1: bucket_starts monotonicity + total.
    bool mono_ok = true;
    for (size_t b = 0; b < num_buckets; ++b) {
        if (h_bucket_starts[b] > h_bucket_starts[b + 1]) { mono_ok = false; break; }
    }
    bool const total_ok = (h_bucket_starts[num_buckets] == count);

    // Validation 2: per-bucket count matches reference.
    bool counts_ok = true;
    for (size_t b = 0; b < num_buckets; ++b) {
        uint32_t const start = h_bucket_starts[b];
        uint32_t const stop  = h_bucket_starts[b + 1];
        if (stop - start != ref_buckets_keys[b].size()) {
            counts_ok = false; break;
        }
    }

    // Validation 3: every key in bucket b has top-bits == b.
    bool consistent_ok = true;
    for (size_t b = 0; b < num_buckets && consistent_ok; ++b) {
        uint32_t const start = h_bucket_starts[b];
        uint32_t const stop  = h_bucket_starts[b + 1];
        for (uint32_t p = start; p < stop; ++p) {
            uint32_t const got_bucket = (h_part_keys[p] >> top_bit_offset) & mask;
            if (got_bucket != b) { consistent_ok = false; break; }
        }
    }

    // Validation 4: per-bucket multiset equality.
    // Sort both sides and compare. Within bucket, sort by (key, val)
    // to canonicalize.
    bool multiset_ok = true;
    for (size_t b = 0; b < num_buckets && multiset_ok; ++b) {
        uint32_t const start = h_bucket_starts[b];
        uint32_t const stop  = h_bucket_starts[b + 1];
        size_t const n_b = stop - start;

        std::vector<std::pair<uint32_t, uint64_t>> got(n_b);
        std::vector<std::pair<uint32_t, uint64_t>> ref(n_b);
        for (size_t i = 0; i < n_b; ++i) {
            got[i] = { h_part_keys[start + i], h_part_vals[start + i] };
            ref[i] = { static_cast<uint32_t>(ref_buckets_keys[b][i]),
                       ref_buckets_vals[b][i] };
        }
        std::sort(got.begin(), got.end());
        std::sort(ref.begin(), ref.end());
        if (got != ref) { multiset_ok = false; break; }
    }

    if (d_scratch) sycl::free(d_scratch, q);
    sycl::free(d_keys_in, q);
    sycl::free(h_pinned_vals, q);
    sycl::free(h_part_keys, q);
    sycl::free(h_part_vals, q);
    sycl::free(h_bucket_starts, q);

    bool const ok = mono_ok && total_ok && counts_ok && consistent_ok && multiset_ok;
    std::printf("%s  seed=%u n=%llu B=%d  [mono=%d total=%d counts=%d cons=%d multi=%d  %.2fms]\n",
                ok ? "PASS" : "FAIL", seed, (unsigned long long)count, num_top_bits,
                mono_ok, total_ok, counts_ok, consistent_ok, multiset_ok, ms);
    return ok;
}

} // namespace

int main()
{
    auto& q = pos2gpu::sycl_backend::queue();
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    // top_bit_offset chosen so partitioning lands within the
    // distribution of the (unique [0, count)) keys. count <= 2^22
    // so a top_bit_offset of (22 - num_top_bits) hits the actual
    // populated bits — equivalent to bucketing the top of the
    // sort range.
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
