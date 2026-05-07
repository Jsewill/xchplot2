// SortDistributed.hpp — multi-shard radix sort wrapper.
//
// Each shard provides a sycl::queue + its local input/output buffers.
// After the call, each shard's output buffer holds the elements whose
// bucket (defined by the upper bits of the key) belongs to that shard,
// sorted within that range.
//
// Phase 2.0b scope: N == 1 fast-path that delegates to the single-shard
// sort in Sort.cuh. N > 1 throws a clear error pointing at the design
// doc. Phase 2.1+ will land the real distributed radix sort
// (per-shard local count → exchange via host pinned → per-shard local
// sort + scatter to bucket-owner → final sort within bucket).
//
// See docs/multi-gpu-single-plot-alt-bucket-partition.md for the
// design rationale.

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include <sycl/sycl.hpp>

namespace pos2gpu {

// Optional per-shard pool used to keep multi-GB pinned host bounces
// resident across the three distributed-sort calls per plot. Forward-
// declared here to avoid pulling host/MultiGpuShardBufferPool.hpp into
// every gpu-layer translation unit.
class ShardBufferPool;

// Selects how the distributed sort moves data across shards.
//
//   HostBounce — (default, Phase 2.1+) D2H every source's input into
//     pinned host buffers, host-side scatter walk to bin items by
//     destination, H2D into receivers, per-receiver local sort.
//     Two PCIe traversals + a host walk. Topology-agnostic: works on
//     any (src, dst) queue pair regardless of peer access.
//
//   Peer — (Phase 2.4b) per-source GPU scatter into per-destination
//     contiguous staging on the source device, then direct D2D memcpy
//     per (src, dst) pair. On NVLink-capable topologies this skips
//     the host-bounce PCIe round-trip entirely; on PCIe-only hosts
//     CUDA's driver typically still routes through host but with
//     fewer copies. Atomic-scatter on GPU produces non-deterministic
//     tie order — output is multiset-equivalent to HostBounce, not
//     byte-identical at same-key tie boundaries.
//
// Both transports are correct; the choice is purely performance.
// Falls back automatically: shards.size() <= 1 always uses the
// single-shard fast path regardless of the requested transport.
enum class DistributedSortTransport {
    HostBounce,
    Peer,
};

// Per-shard handles for the distributed sort. The caller owns the
// queue and the input/output buffers; the wrapper coordinates the
// cross-shard exchange (Phase 2.1+) without taking ownership.
//
// Same in/out ping-pong contract as the single-shard sort:
// keys_in / vals_in are scratch on input — they get clobbered. The
// result lands in keys_out / vals_out.
struct DistributedSortPairsShard {
    sycl::queue* queue;

    std::uint32_t* keys_in;
    std::uint32_t* vals_in;
    std::uint64_t  count;          // input element count for this shard

    std::uint32_t* keys_out;
    std::uint32_t* vals_out;
    std::uint64_t  out_capacity;   // max elements this shard's output can hold

    // Filled by the implementation post-sort:
    std::uint64_t  out_count;      // actual elements placed in this shard's output

    // Optional. When non-null, the host-bounce pinned buffers and the
    // per-receiver sort scratch are routed through this pool's
    // ensure_host / ensure_bytes slots so consecutive distributed-sort
    // calls (T1 → T2 → T3 in a single plot) and consecutive plots reuse
    // the same page-locked / device allocations.
    ShardBufferPool* pool = nullptr;
};

struct DistributedSortKeysU64Shard {
    sycl::queue* queue;

    std::uint64_t* keys_in;
    std::uint64_t  count;

    std::uint64_t* keys_out;
    std::uint64_t  out_capacity;

    std::uint64_t  out_count;

    ShardBufferPool* pool = nullptr;
};

// Pairs sort with uint32 key and uint64 value. Used by T1's distributed
// post-match sort, where the key is the 32-bit match_info and the value
// is the 64-bit T1 meta. Same ping-pong contract as the u32/u32 form:
// keys_in / vals_in are scratch on input, results land in keys_out /
// vals_out.
struct DistributedSortPairsU32U64Shard {
    sycl::queue* queue;

    std::uint32_t* keys_in;
    std::uint64_t* vals_in;
    std::uint64_t  count;

    std::uint32_t* keys_out;
    std::uint64_t* vals_out;
    std::uint64_t  out_capacity;

    std::uint64_t  out_count;

    ShardBufferPool* pool = nullptr;
};

// Pairs sort with uint32 key and (uint64, uint32) value pair. Used by
// T2's distributed post-match sort: the key is match_info, the values
// are the T2 meta (u64) and x_bits (u32) streams. Same ping-pong
// contract: every keys_in / vals_*_in buffer is scratch on input.
struct DistributedSortPairsU32U64U32Shard {
    sycl::queue* queue;

    std::uint32_t* keys_in;
    std::uint64_t* vals_a_in;     // meta
    std::uint32_t* vals_b_in;     // x_bits
    std::uint64_t  count;

    std::uint32_t* keys_out;
    std::uint64_t* vals_a_out;
    std::uint32_t* vals_b_out;
    std::uint64_t  out_capacity;

    std::uint64_t  out_count;

    ShardBufferPool* pool = nullptr;
};

// Sort (key, value) pairs by uint32 key over [begin_bit, end_bit) bits,
// distributed across the shards.
//
// Two-call sizing-then-sort contract mirrors CUB:
//   - First call with d_temp_storage == nullptr fills temp_bytes with
//     the required scratch size (per-shard allocation; caller owns).
//     Phase 2.0b: same as single-shard temp_bytes (N=1 fast-path).
//   - Second call with d_temp_storage != nullptr performs the sort.
//     Each shard's output bucket range is determined by the wrapper.
//
// On N=1, behaves identically to launch_sort_pairs_u32_u32 against the
// sole shard's data. On N>1, throws std::runtime_error in Phase 2.0b.
void launch_sort_pairs_u32_u32_distributed(
    void* d_temp_storage,
    std::size_t& temp_bytes,
    std::vector<DistributedSortPairsShard>& shards,
    int begin_bit, int end_bit,
    DistributedSortTransport transport = DistributedSortTransport::HostBounce);

// Distributed analogue of launch_sort_keys_u64.
void launch_sort_keys_u64_distributed(
    void* d_temp_storage,
    std::size_t& temp_bytes,
    std::vector<DistributedSortKeysU64Shard>& shards,
    int begin_bit, int end_bit,
    DistributedSortTransport transport = DistributedSortTransport::HostBounce);

// Sort (uint32 key, uint64 value) pairs by uint32 key over
// [begin_bit, end_bit) bits, distributed across the shards. Same
// sizing-then-sort contract as launch_sort_pairs_u32_u32_distributed.
//
// N=1 fast path: identity-index radix sort + 64-bit gather, mirroring
// the single-GPU T1-sort pattern in GpuPipeline.cpp. Caller-owned
// temp_bytes covers the underlying u32/u32 sort scratch + 2 * count *
// sizeof(uint32) for the identity / sorted-index buffers.
//
// N>=2: same host-pinned bounce as the u32/u32 form, with u64 values
// carried alongside the key on the scatter walk; each receiver runs a
// local identity-index sort + gather to reconstruct the sorted vals_out.
// Per-receiver scratch is allocated internally (temp_bytes = 0).
void launch_sort_pairs_u32_u64_distributed(
    void* d_temp_storage,
    std::size_t& temp_bytes,
    std::vector<DistributedSortPairsU32U64Shard>& shards,
    int begin_bit, int end_bit,
    DistributedSortTransport transport = DistributedSortTransport::HostBounce);

// Sort (uint32 key, uint64 value_a, uint32 value_b) triples by uint32
// key over [begin_bit, end_bit) bits, distributed across the shards.
// Same sizing-then-sort contract as the u32/u32 form.
//
// N=1 fast path: identity-index radix sort + launch_permute_t2 to gather
// both vals streams in one fused kernel (mirrors GpuPipeline.cpp's T2
// sort). Caller-owned temp_bytes covers the underlying u32/u32 sort
// scratch + 2 * count * sizeof(uint32) for the identity / sorted-index
// buffers.
//
// N>=2: same host-pinned bounce as the u32/u32 form, with the u64 + u32
// vals carried alongside the key on the scatter walk; each receiver
// runs a local identity-index sort + launch_permute_t2 to reconstruct
// vals_a_out / vals_b_out. Per-receiver scratch is allocated internally
// (temp_bytes = 0).
void launch_sort_pairs_u32_u64u32_distributed(
    void* d_temp_storage,
    std::size_t& temp_bytes,
    std::vector<DistributedSortPairsU32U64U32Shard>& shards,
    int begin_bit, int end_bit,
    DistributedSortTransport transport = DistributedSortTransport::HostBounce);

} // namespace pos2gpu
