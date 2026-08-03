// SpillTileReader.hpp — the callback view a streaming-partition pass uses to
// pull its source tiles off disk.
//
// This is a plain POD with two function pointers and it lives in its own header
// for one reason: it is the ONLY thing the spill engine needs from the GPU
// side. StreamingPartition.cuh includes <sycl/sycl.hpp>, so leaving the struct
// there would have forced SpillEngine.hpp to drag SYCL in with it, and a
// SYCL-free SpillEngine.hpp is what lets spill_engine_test exercise the ticket
// protocol on a machine with no GPU at all. Both sides include this instead.
//
// P1.5 host-RAM disk-offload: when a non-null SpillTileReader is passed to a
// streaming partition, pass 2 does NOT read from the host array (which is then
// null); instead each source-values tile is pread from disk by a background
// I/O worker into one of TWO ping-pong windows, double-buffered so tile t+1's
// pread overlaps tile t's partition kernel. The primitive drives it:
//   submit(ctx, slot, off_bytes, bytes)  — enqueue an async pread of `bytes`
//       at file offset `off_bytes` into window `slot` (0/1).
//   wait(ctx, slot)                      — block until window `slot`'s most
//       recent submit has completed.
// win[slot] is the pinned destination window; each tile must be <= win_entries
// u64 entries (the caller sizes tile_count to guarantee it). A null reader
// keeps the in-memory path, byte-identical.
//
// Slot discipline differs by variant, and the engine only guarantees that a
// window holds what its LAST submit put there:
//   single-value (u32_u64): one stream ping-pongs slots 0/1 via `slot ^ 1`, so
//                           tile t+1 loads while tile t partitions.
//   triple (u32_u64_u32):   a tile needs BOTH streams resident at once, so it
//                           holds both windows and has no cross-tile prefetch.
//                           Widening to four windows to buy one was measured
//                           and did not pay — see kNumWindows in
//                           SpillEngine.hpp.

#pragma once

#include <cstdint>

namespace pos2gpu {

struct SpillTileReader {
    void*     ctx         = nullptr;
    uint64_t* win[2]      = {nullptr, nullptr};
    uint64_t  win_entries = 0;
    bool      overlap     = true;   // false => synchronous per-tile (measurement only)
    void (*submit)(void* ctx, int slot, uint64_t off_bytes, uint64_t bytes) = nullptr;
    void (*wait)(void* ctx, int slot)                                       = nullptr;
};

}  // namespace pos2gpu
