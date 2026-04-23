// XsKernel.cpp — orchestrates Xs construction on a SYCL queue.
//
// Pipeline:
//   1. launch_xs_gen:  writes (g(x⊕xor_const), x) into (keys_a, vals_a).
//   2. launch_sort_pairs_u32_u32: stable radix sort by the bottom k bits.
//   3. launch_xs_pack: fold sorted (keys, vals) into XsCandidateGpu[total].
//
// All scratch is allocated by the caller; on the first call with
// d_temp_storage == nullptr the function only writes the required
// *temp_bytes and returns without launching anything.

#include "gpu/AesHashGpu.cuh"
#include "gpu/Sort.cuh"
#include "gpu/XsKernel.cuh"
#include "gpu/XsKernels.cuh"

#include <sycl/sycl.hpp>

#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace pos2gpu {

namespace {

// Mirrors pos2-chip/src/pos/ProofConstants.hpp:14
constexpr uint32_t kTestnetGXorConst = 0xA3B1C4D7u;

// Layout of caller-provided d_temp_storage:
//   [0                  .. cub_bytes)            CUB sort scratch
//   [keys_a_off         .. keys_a_off + N*4)     keys_a (uint32)  (*)
//   [keys_b_off         .. keys_b_off + N*4)     keys_b (uint32)
//   [vals_a_off         .. vals_a_off + N*4)     vals_a (uint32)
//   [vals_b_off         .. vals_b_off + N*4)     vals_b (uint32)
// (*) In split mode (split_keys_a != nullptr) the keys_a slot is OMITTED
// from d_temp_storage — keys_a_off is set to SIZE_MAX as a sentinel and
// keys_b_off follows directly after cub_scratch. Total bytes drop by
// one aligned (N*u32) block (~1 GiB at k=28).
struct ScratchLayout {
    size_t cub_bytes;
    size_t keys_a_off;
    size_t keys_b_off;
    size_t vals_a_off;
    size_t vals_b_off;
    size_t total_bytes;
};

inline size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

ScratchLayout layout_for(uint64_t total, size_t cub_bytes, bool split_keys_a)
{
    ScratchLayout s{};
    s.cub_bytes  = cub_bytes;
    size_t cur   = align_up(s.cub_bytes, 256);
    if (split_keys_a) {
        s.keys_a_off = ~size_t{0};  // sentinel: keys_a lives externally
    } else {
        s.keys_a_off = cur; cur += sizeof(uint32_t) * total; cur = align_up(cur, 256);
    }
    s.keys_b_off = cur; cur += sizeof(uint32_t) * total; cur = align_up(cur, 256);
    s.vals_a_off = cur; cur += sizeof(uint32_t) * total; cur = align_up(cur, 256);
    s.vals_b_off = cur; cur += sizeof(uint32_t) * total; cur = align_up(cur, 256);
    s.total_bytes = cur;
    return s;
}

} // namespace

void launch_construct_xs(
    uint8_t const* plot_id_bytes, int k, bool testnet,
    XsCandidateGpu* d_out, void* d_temp_storage, size_t* temp_bytes,
    sycl::queue& q, void* split_keys_a)
{
    return launch_construct_xs_profiled(plot_id_bytes, k, testnet,
                                        d_out, d_temp_storage, temp_bytes,
                                        nullptr, nullptr, q, split_keys_a);
}

void launch_construct_xs_profiled(
    uint8_t const* plot_id_bytes,
    int k,
    bool testnet,
    XsCandidateGpu* d_out,
    void* d_temp_storage,
    size_t* temp_bytes,
    cudaEvent_t /*after_gen*/,
    cudaEvent_t /*after_sort*/,
    sycl::queue& q,
    void* split_keys_a)
{
    // NOTE: the cudaEvent_t after_gen / after_sort parameters are kept
    // for API compatibility but no longer recorded. xs_bench's per-phase
    // timing is therefore zero through this call; use chrono on the host
    // around launch_construct_xs to measure end-to-end wall time. A
    // sycl::event-based profiling overload is the natural follow-up.

    if (k < 18 || k > 32 || (k & 1) != 0) throw std::invalid_argument("invalid argument to launch wrapper");
    if (!plot_id_bytes || !temp_bytes)    throw std::invalid_argument("invalid argument to launch wrapper");

    uint64_t const total = 1ULL << k;

    // Query CUB temp size via the wrapper (sizing mode: null storage).
    size_t cub_bytes = 0;
    launch_sort_pairs_u32_u32(
        nullptr, cub_bytes,
        nullptr, nullptr,
        nullptr, nullptr,
        total, /*begin_bit=*/0, /*end_bit=*/k, q);

    bool const split = (split_keys_a != nullptr);
    auto sl = layout_for(total, cub_bytes, split);

    if (d_temp_storage == nullptr) {
        *temp_bytes = sl.total_bytes;

        return;
    }
    if (*temp_bytes < sl.total_bytes) throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_out)                       throw std::invalid_argument("invalid argument to launch wrapper");

    auto* base = static_cast<uint8_t*>(d_temp_storage);
    auto* cub_scratch = base; // first cub_bytes
    auto* keys_a = split
        ? static_cast<uint32_t*>(split_keys_a)
        : reinterpret_cast<uint32_t*>(base + sl.keys_a_off);
    auto* keys_b = reinterpret_cast<uint32_t*>(base + sl.keys_b_off);
    auto* vals_a = reinterpret_cast<uint32_t*>(base + sl.vals_a_off);
    auto* vals_b = reinterpret_cast<uint32_t*>(base + sl.vals_b_off);

    AesHashKeys keys = make_keys(plot_id_bytes);
    uint32_t xor_const = testnet ? kTestnetGXorConst : 0u;

    // Sub-phase wall-time breakdown — useful when GpuPipeline's outer
    // "Xs gen+sort" phase dominates total wall (notably on the SYCL/HIP
    // backend, where the Xs phase has been observed at ~40% on RDNA2 vs
    // ~6% on NVIDIA). Gated on POS2GPU_PHASE_TIMING=1 so the q.wait()s
    // don't perturb production runs.
    bool const xs_timing = [] {
        char const* v = std::getenv("POS2GPU_PHASE_TIMING");
        return v && v[0] == '1';
    }();
    using xs_clock = std::chrono::steady_clock;
    auto xs_now = [&] { return xs_clock::now(); };
    auto xs_elapsed_ms = [&](xs_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(xs_now() - t0).count();
    };
    auto xs_t0 = xs_now();
    if (xs_timing) q.wait();

    // Phase 1: generate (match_info, x) into keys_a / vals_a
    launch_xs_gen(keys, keys_a, vals_a, total, k, xor_const, q);
    double t_gen = 0.0;
    if (xs_timing) { q.wait(); t_gen = xs_elapsed_ms(xs_t0); xs_t0 = xs_now(); }

    // Phase 2: stable radix sort by (key low k bits) — keys_a → keys_b,
    // vals_a → vals_b. (We give up CUB's DoubleBuffer optimisation here,
    // costing one extra pass at most; pack reads from the b side.)
    launch_sort_pairs_u32_u32(
        cub_scratch, cub_bytes,
        keys_a, keys_b,
        vals_a, vals_b,
        total, /*begin_bit=*/0, /*end_bit=*/k, q);
    double t_sort = 0.0;
    if (xs_timing) { q.wait(); t_sort = xs_elapsed_ms(xs_t0); xs_t0 = xs_now(); }

    // Phase 3: pack the sorted side into AoS XsCandidateGpu in d_out.
    launch_xs_pack(keys_b, vals_b, d_out, total, q);
    double t_pack = 0.0;
    if (xs_timing) { q.wait(); t_pack = xs_elapsed_ms(xs_t0); }

    if (xs_timing) {
        double const total_ms = t_gen + t_sort + t_pack;
        std::fprintf(stderr,
            "[xs-timing] gen=%.1fms(%.0f%%) sort=%.1fms(%.0f%%) pack=%.1fms(%.0f%%) total=%.1fms\n",
            t_gen,  total_ms > 0.0 ? 100.0 * t_gen  / total_ms : 0.0,
            t_sort, total_ms > 0.0 ? 100.0 * t_sort / total_ms : 0.0,
            t_pack, total_ms > 0.0 ? 100.0 * t_pack / total_ms : 0.0,
            total_ms);
    }
}

} // namespace pos2gpu
