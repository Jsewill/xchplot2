// XsKernel.cu — implementation of launch_construct_xs.
//
// Pipeline:
//   1. Phase 1 kernel writes XsCandidateGpu[x] = { g(x), x } for x in [0, 2^k).
//   2. Pack into (key=match_info, value=x) and call cub::DeviceRadixSort::
//      SortPairs over the bottom k bits. CUB's radix sort is stable
//      (preserves relative order for equal keys), matching pos2-chip's
//      RadixSort which is multi-pass LSD radix.
//   3. Repack sorted (key, value) back into XsCandidateGpu in d_out.
//
// All scratch is allocated by the caller; on first call with d_temp_storage
// == nullptr the function only writes the required *temp_bytes and returns
// without launching anything.

#include "gpu/AesGpu.cuh"
#include "gpu/AesHashGpu.cuh"
#include "gpu/XsKernel.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdint>

namespace pos2gpu {

namespace {

// Mirrors pos2-chip/src/pos/ProofConstants.hpp:14
constexpr uint32_t kTestnetGXorConst = 0xA3B1C4D7u;

__global__ void gen_kernel(
    AesHashKeys keys,
    uint32_t* __restrict__ keys_out, // match_info
    uint32_t* __restrict__ vals_out, // x
    uint64_t total,
    int k,
    uint32_t xor_const)
{
    __shared__ uint32_t sT[4 * 256];
    load_aes_tables_smem(sT);
    __syncthreads();

    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    uint32_t x = static_cast<uint32_t>(idx);
    uint32_t mixed = x ^ xor_const;
    keys_out[idx] = g_x_smem(keys, mixed, k, sT, kAesGRounds);
    vals_out[idx] = x;
}

__global__ void pack_kernel(
    uint32_t const* __restrict__ keys_in,
    uint32_t const* __restrict__ vals_in,
    XsCandidateGpu* __restrict__ out,
    uint64_t total)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    out[idx] = XsCandidateGpu{ keys_in[idx], vals_in[idx] };
}

// Layout of caller-provided d_temp_storage (single arena):
//
//   [0                  .. keys_in_off)             reserved for CUB scratch
//   [keys_in_off        .. keys_in_off + N*4)        keys_in   (uint32)
//   [keys_out_off       .. keys_out_off + N*4)       keys_out  (uint32)
//   [vals_in_off        .. vals_in_off + N*4)        vals_in   (uint32)
//   [vals_out_off       .. vals_out_off + N*4)       vals_out  (uint32)
//
// CUB SortPairs alternates ping-pong between in/out; we use the
// `DoubleBuffer` API to let CUB pick which side ends up holding the
// sorted result.

struct ScratchLayout {
    size_t cub_bytes;     // bytes for CUB's own scratch
    size_t keys_a_off;    // offset to keys buffer A
    size_t keys_b_off;    // offset to keys buffer B
    size_t vals_a_off;    // offset to vals buffer A
    size_t vals_b_off;    // offset to vals buffer B
    size_t total_bytes;
};

constexpr size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

ScratchLayout layout_for(uint64_t total, size_t cub_bytes)
{
    ScratchLayout s{};
    s.cub_bytes = cub_bytes;
    size_t cur = align_up(s.cub_bytes, 256);
    s.keys_a_off = cur; cur += sizeof(uint32_t) * total; cur = align_up(cur, 256);
    s.keys_b_off = cur; cur += sizeof(uint32_t) * total; cur = align_up(cur, 256);
    s.vals_a_off = cur; cur += sizeof(uint32_t) * total; cur = align_up(cur, 256);
    s.vals_b_off = cur; cur += sizeof(uint32_t) * total; cur = align_up(cur, 256);
    s.total_bytes = cur;
    return s;
}

} // namespace

cudaError_t launch_construct_xs(
    uint8_t const* plot_id_bytes, int k, bool testnet,
    XsCandidateGpu* d_out, void* d_temp_storage, size_t* temp_bytes,
    cudaStream_t stream)
{
    return launch_construct_xs_profiled(plot_id_bytes, k, testnet,
                                        d_out, d_temp_storage, temp_bytes,
                                        nullptr, nullptr, stream);
}

// Public sub-kernel launchers used by the streaming path's inline
// gen+sort+pack sequence (avoids the bundled d_temp_storage blob that
// otherwise forces keys_a + vals_a to stay alive through pack).
cudaError_t launch_xs_gen(
    uint8_t const* plot_id_bytes, int k, bool testnet,
    uint32_t* d_keys_out, uint32_t* d_vals_out,
    cudaStream_t stream)
{
    if (k < 18 || k > 32 || (k & 1) != 0) return cudaErrorInvalidValue;
    if (!plot_id_bytes || !d_keys_out || !d_vals_out) return cudaErrorInvalidValue;

    uint64_t const total = 1ULL << k;
    AesHashKeys keys = make_keys(plot_id_bytes);
    uint32_t xor_const = testnet ? kTestnetGXorConst : 0u;

    constexpr int kThreads = 256;
    uint64_t blocks_u64 = (total + kThreads - 1) / kThreads;
    if (blocks_u64 > UINT_MAX) return cudaErrorInvalidValue;
    unsigned blocks = static_cast<unsigned>(blocks_u64);

    gen_kernel<<<blocks, kThreads, 0, stream>>>(
        keys, d_keys_out, d_vals_out, total, k, xor_const);
    return cudaGetLastError();
}

cudaError_t launch_xs_pack(
    uint32_t const* d_keys_in, uint32_t const* d_vals_in,
    XsCandidateGpu* d_out, uint64_t total,
    cudaStream_t stream)
{
    if (!d_keys_in || !d_vals_in || !d_out) return cudaErrorInvalidValue;

    constexpr int kThreads = 256;
    uint64_t blocks_u64 = (total + kThreads - 1) / kThreads;
    if (blocks_u64 > UINT_MAX) return cudaErrorInvalidValue;
    unsigned blocks = static_cast<unsigned>(blocks_u64);

    pack_kernel<<<blocks, kThreads, 0, stream>>>(
        d_keys_in, d_vals_in, d_out, total);
    return cudaGetLastError();
}

cudaError_t launch_construct_xs_profiled(
    uint8_t const* plot_id_bytes,
    int k,
    bool testnet,
    XsCandidateGpu* d_out,
    void* d_temp_storage,
    size_t* temp_bytes,
    cudaEvent_t after_gen,
    cudaEvent_t after_sort,
    cudaStream_t stream)
{
    if (k < 18 || k > 32 || (k & 1) != 0) return cudaErrorInvalidValue;
    if (!plot_id_bytes || !temp_bytes)    return cudaErrorInvalidValue;

    uint64_t const total = 1ULL << k;

    // Query CUB temp size once (depends only on N).
    cub::DoubleBuffer<uint32_t> probe_keys(nullptr, nullptr);
    cub::DoubleBuffer<uint32_t> probe_vals(nullptr, nullptr);
    size_t cub_bytes = 0;
    cudaError_t err = cub::DeviceRadixSort::SortPairs(
        nullptr, cub_bytes,
        probe_keys, probe_vals,
        total, /*begin_bit=*/0, /*end_bit=*/k, stream);
    if (err != cudaSuccess) return err;

    auto sl = layout_for(total, cub_bytes);

    if (d_temp_storage == nullptr) {
        *temp_bytes = sl.total_bytes;
        return cudaSuccess;
    }
    if (*temp_bytes < sl.total_bytes) return cudaErrorInvalidValue;
    if (!d_out)                       return cudaErrorInvalidValue;

    auto* base = static_cast<uint8_t*>(d_temp_storage);
    auto* cub_scratch = base; // first cub_bytes
    auto* keys_a = reinterpret_cast<uint32_t*>(base + sl.keys_a_off);
    auto* keys_b = reinterpret_cast<uint32_t*>(base + sl.keys_b_off);
    auto* vals_a = reinterpret_cast<uint32_t*>(base + sl.vals_a_off);
    auto* vals_b = reinterpret_cast<uint32_t*>(base + sl.vals_b_off);

    AesHashKeys keys = make_keys(plot_id_bytes);
    uint32_t xor_const = testnet ? kTestnetGXorConst : 0u;

    constexpr int kThreads = 256;
    uint64_t blocks_u64 = (total + kThreads - 1) / kThreads;
    if (blocks_u64 > UINT_MAX) return cudaErrorInvalidValue;
    unsigned blocks = static_cast<unsigned>(blocks_u64);

    // Phase 1: generate (match_info, x) into keys_a / vals_a
    gen_kernel<<<blocks, kThreads, 0, stream>>>(keys, keys_a, vals_a, total, k, xor_const);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    if (after_gen) cudaEventRecord(after_gen, stream);

    // Phase 2: stable radix sort by (key low k bits)
    cub::DoubleBuffer<uint32_t> keys_buf(keys_a, keys_b);
    cub::DoubleBuffer<uint32_t> vals_buf(vals_a, vals_b);
    err = cub::DeviceRadixSort::SortPairs(
        cub_scratch, cub_bytes,
        keys_buf, vals_buf,
        total, /*begin_bit=*/0, /*end_bit=*/k, stream);
    if (err != cudaSuccess) return err;

    // Phase 3: pack the side CUB ended up writing into d_out
    pack_kernel<<<blocks, kThreads, 0, stream>>>(
        keys_buf.Current(), vals_buf.Current(), d_out, total);
    if (after_sort) cudaEventRecord(after_sort, stream);
    return cudaGetLastError();
}

} // namespace pos2gpu
