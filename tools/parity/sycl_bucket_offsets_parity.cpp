// sycl_bucket_offsets_parity — SYCL port of compute_bucket_offsets
// (src/gpu/T1Kernel.cu:58) verified against a CPU reference on synthetic
// input. First slice of the SYCL backend port: proves the AdaptiveCpp
// toolchain works end-to-end before we touch the production pipeline.
//
// The kernel is "for each bucket b in [0, num_buckets), find the lowest
// index i in `sorted` such that (sorted[i].match_info >> shift) >= b" —
// one thread per bucket runs a binary search and writes offsets[b].
// Thread num_buckets writes the sentinel offsets[num_buckets] = total.
//
// Synthetic input: a sorted random XsCandidateGpu[] with match_info
// drawn uniformly from [0, num_buckets << shift) so every bucket is
// non-trivially populated. Reference is std::lower_bound on the same
// shifted key. Pass criterion: byte-for-byte memcmp of offsets[].

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

// Local copy of pos2gpu::XsCandidateGpu — keeps this TU free of the
// CUDA-laden gpu/XsKernel.cuh include chain. Layout-checked below.
struct XsCandidateGpu {
    uint32_t match_info;
    uint32_t x;
};
static_assert(sizeof(XsCandidateGpu) == 8, "must match pos2-chip Xs_Candidate layout");

std::vector<XsCandidateGpu> make_sorted_input(uint64_t total, uint64_t value_range, uint32_t seed)
{
    std::mt19937_64 rng(seed);
    std::vector<XsCandidateGpu> v(total);
    for (uint64_t i = 0; i < total; ++i) {
        v[i].match_info = static_cast<uint32_t>(rng() % value_range);
        v[i].x          = static_cast<uint32_t>(rng());
    }
    std::sort(v.begin(), v.end(),
              [](XsCandidateGpu const& a, XsCandidateGpu const& b) {
                  return a.match_info < b.match_info;
              });
    return v;
}

std::vector<uint64_t> reference_offsets(
    std::vector<XsCandidateGpu> const& sorted,
    int num_match_target_bits,
    uint32_t num_buckets)
{
    std::vector<uint64_t> offsets(num_buckets + 1);
    uint32_t const shift = static_cast<uint32_t>(num_match_target_bits);
    uint64_t const total = sorted.size();
    for (uint32_t b = 0; b < num_buckets; ++b) {
        uint64_t lo = 0, hi = total;
        while (lo < hi) {
            uint64_t mid = lo + ((hi - lo) >> 1);
            uint32_t v   = sorted[mid].match_info >> shift;
            if (v < b) lo = mid + 1;
            else       hi = mid;
        }
        offsets[b] = lo;
    }
    offsets[num_buckets] = total;
    return offsets;
}

std::vector<uint64_t> sycl_offsets(
    sycl::queue& q,
    std::vector<XsCandidateGpu> const& sorted,
    int num_match_target_bits,
    uint32_t num_buckets)
{
    uint64_t const total     = sorted.size();
    size_t   const out_count = static_cast<size_t>(num_buckets) + 1;
    constexpr size_t threads = 256;
    size_t   const groups    = (out_count + threads - 1) / threads;

    XsCandidateGpu* d_sorted  = sycl::malloc_device<XsCandidateGpu>(total, q);
    uint64_t*       d_offsets = sycl::malloc_device<uint64_t>(out_count, q);

    q.memcpy(d_sorted, sorted.data(), sizeof(XsCandidateGpu) * total).wait();

    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=](sycl::nd_item<1> it) {
            uint32_t b = static_cast<uint32_t>(it.get_global_id(0));
            if (b > num_buckets) return;
            if (b == num_buckets) { d_offsets[num_buckets] = total; return; }

            uint32_t bucket_shift = static_cast<uint32_t>(num_match_target_bits);
            uint64_t lo = 0, hi = total;
            while (lo < hi) {
                uint64_t mid = lo + ((hi - lo) >> 1);
                uint32_t v   = d_sorted[mid].match_info >> bucket_shift;
                if (v < b) lo = mid + 1;
                else       hi = mid;
            }
            d_offsets[b] = lo;
        }).wait();

    std::vector<uint64_t> out(out_count);
    q.memcpy(out.data(), d_offsets, sizeof(uint64_t) * out_count).wait();

    sycl::free(d_sorted, q);
    sycl::free(d_offsets, q);
    return out;
}

bool run_for(sycl::queue& q, uint32_t seed, uint64_t total,
             int num_match_target_bits, uint32_t num_buckets)
{
    uint64_t const value_range = uint64_t(num_buckets) << num_match_target_bits;
    auto sorted    = make_sorted_input(total, value_range, seed);
    auto reference = reference_offsets(sorted, num_match_target_bits, num_buckets);
    auto actual    = sycl_offsets(q, sorted, num_match_target_bits, num_buckets);

    if (std::memcmp(reference.data(), actual.data(),
                    sizeof(uint64_t) * reference.size()) == 0) {
        std::printf("PASS  seed=%u total=%llu shift=%d buckets=%u\n",
                    seed, (unsigned long long)total,
                    num_match_target_bits, num_buckets);
        return true;
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        if (reference[i] != actual[i]) {
            std::fprintf(stderr,
                "FAIL  seed=%u  bucket=%zu  ref=%llu  actual=%llu\n",
                seed, i,
                (unsigned long long)reference[i],
                (unsigned long long)actual[i]);
            break;
        }
    }
    return false;
}

} // namespace

int main()
{
    sycl::queue q{ sycl::default_selector_v };
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    // Sizes representative of T1 at small k (slice 1 is correctness, not perf).
    // num_buckets = num_sections (4) * num_match_keys (4) = 16 for k<28.
    struct Case { uint64_t total; int shift; uint32_t buckets; };
    Case const cases[] = {
        { 1ull << 18, 14, 16 },   // k=18
        { 1ull << 20, 16, 16 },   // k=20
        { 1ull << 22, 18, 16 },   // k=22
        { 1ull << 24, 20, 16 },   // k=24
    };

    bool all_pass = true;
    for (uint32_t seed : { 1u, 7u, 31u }) {
        for (auto const& c : cases) {
            if (!run_for(q, seed, c.total, c.shift, c.buckets)) all_pass = false;
        }
    }
    return all_pass ? 0 : 1;
}
