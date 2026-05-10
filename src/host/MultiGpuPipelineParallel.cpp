// MultiGpuPipelineParallel.cpp — Phase 2.1c/d orchestrator. See header.

#include "host/MultiGpuPipelineParallel.hpp"

#include "gpu/SyclBackend.hpp"
#include "host/HostPinnedPool.hpp"
#include "host/PoolSizing.hpp"

#include <sycl/sycl.hpp>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>

namespace pos2gpu {

namespace {

std::uint64_t default_vram_for_device(int id)
{
    auto const& devs = sycl_backend::usable_gpu_devices();
    if (id < 0 || static_cast<std::size_t>(id) >= devs.size()) {
        throw std::runtime_error(
            "select_pipeline_devices: device id " + std::to_string(id) +
            " out of range (have " + std::to_string(devs.size()) + " GPUs)");
    }
    return devs[id].get_info<sycl::info::device::global_mem_size>();
}

} // namespace

PipelineDeviceAssignment select_pipeline_devices(
    std::vector<int> const&                  device_ids,
    std::function<std::uint64_t(int)> const& vram_for_device)
{
    if (device_ids.size() != 2) {
        throw std::runtime_error(
            "select_pipeline_devices: exactly 2 device ids required (got " +
            std::to_string(device_ids.size()) + "); N-stage in progress");
    }
    int const dev_a = device_ids[0];
    int const dev_b = device_ids[1];
    if (dev_a < 0 || dev_b < 0) {
        throw std::runtime_error(
            "select_pipeline_devices: device ids must be non-negative");
    }
    auto const a_vram = vram_for_device(dev_a);
    auto const b_vram = vram_for_device(dev_b);
    PipelineDeviceAssignment out;
    if (a_vram >= b_vram) {
        out.dev_first             = dev_a;
        out.dev_second            = dev_b;
        out.dev_first_vram_bytes  = a_vram;
        out.dev_second_vram_bytes = b_vram;
        out.reordered             = false;
    } else {
        out.dev_first             = dev_b;
        out.dev_second            = dev_a;
        out.dev_first_vram_bytes  = b_vram;
        out.dev_second_vram_bytes = a_vram;
        out.reordered             = true;
    }
    return out;
}

PipelineDeviceAssignment select_pipeline_devices(std::vector<int> const& device_ids)
{
    return select_pipeline_devices(device_ids, default_vram_for_device);
}

namespace {

// Buffer set for the T2 boundary. Allocated on the orchestrator
// thread; both worker threads read/write through these pointers. The
// first device's queue is used for the malloc_host so the buffers
// land in a shared pinned-host pool — on CUDA that's portable across
// all devices in the same process by default.
//
// Portability caveat (Phase 2-A hypothesis): AdaptiveCpp's CUDA
// backend is *expected* to allocate with cudaHostAllocPortable so the
// host pages are mapped into every device context. Validated on Ada
// Lovelace (RTX 4000 Ada). Observed once on Ampere (2× A4000) the
// first plot segfaulted and left GPU 0 in nvidia-smi ERR! state — if
// future investigation confirms the cause is non-portable pinned
// memory on Ampere drivers, the fix is to either (a) interop with the
// CUDA backend to call cudaHostRegister with cudaHostRegisterPortable
// after allocation, or (b) allocate one buffer set per device and add
// a host-staging copy at the boundary. Neither is needed if the
// upstream allocator already passes the portable flag.
struct BoundaryBuffers {
    std::uint64_t* pinned_dst        = nullptr;
    std::uint64_t* h_meta            = nullptr;
    std::uint64_t* h_t2_meta         = nullptr;
    std::uint32_t* h_t2_xbits        = nullptr;
    std::uint32_t* h_keys_merged     = nullptr;
    sycl::queue*   alloc_queue       = nullptr;
};

void free_boundary(BoundaryBuffers& b)
{
    if (!b.alloc_queue) return;
    if (b.pinned_dst)    sycl::free(b.pinned_dst,    *b.alloc_queue);
    if (b.h_meta)        sycl::free(b.h_meta,        *b.alloc_queue);
    if (b.h_t2_meta)     sycl::free(b.h_t2_meta,     *b.alloc_queue);
    if (b.h_t2_xbits)    sycl::free(b.h_t2_xbits,    *b.alloc_queue);
    if (b.h_keys_merged) sycl::free(b.h_keys_merged, *b.alloc_queue);
    b.pinned_dst = nullptr;
    b.h_meta = b.h_t2_meta = nullptr;
    b.h_t2_xbits = nullptr;
    b.h_keys_merged = nullptr;
}

} // namespace

PipelineParallelSplitResult run_pipeline_parallel_split(
    GpuPipelineConfig const&              cfg,
    std::vector<int> const&               device_ids,
    std::vector<PipelineStageTier> const& tiers)
{
    if (cfg.k < 18 || cfg.k > 32 || (cfg.k & 1) != 0) {
        throw std::runtime_error("k must be even in [18, 32]");
    }
    if (cfg.strength < 2) {
        throw std::runtime_error("strength must be >= 2");
    }
    if (device_ids.size() != 2) {
        throw std::runtime_error(
            "run_pipeline_parallel_split: exactly 2 device ids required (got " +
            std::to_string(device_ids.size()) + "); N-stage in progress");
    }
    if (!tiers.empty() && tiers.size() != device_ids.size()) {
        throw std::runtime_error(
            "run_pipeline_parallel_split: tiers must be empty or match "
            "device_ids size");
    }
    int const device_first  = device_ids[0];
    int const device_second = device_ids[1];
    PipelineStageTier const tier_first =
        tiers.empty() ? PipelineStageTier::Tiny : tiers[0];
    PipelineStageTier const tier_second =
        tiers.empty() ? PipelineStageTier::Tiny : tiers[1];
    if (device_first < 0 || device_second < 0) {
        throw std::runtime_error(
            "run_pipeline_parallel_split: device ids must be non-negative");
    }

    int const num_section_bits = (cfg.k < 28) ? 2 : (cfg.k - 26);
    std::uint64_t const cap =
        max_pairs_per_section(cfg.k, num_section_bits) *
        (std::uint64_t{1} << num_section_bits);

    // Allocate the boundary buffers on a queue bound to device_first.
    // bind_current_device(device_first) makes sycl_backend::queue() lazily
    // construct (in this thread) a queue rooted at that device. The
    // buffers land in that queue's host-pinned pool; on CUDA they're
    // portable to other devices in the same process.
    BoundaryBuffers buf;
    {
        bind_current_device(device_first);
        sycl::queue& q = sycl_backend::queue();
        buf.alloc_queue = &q;
        buf.pinned_dst    = static_cast<std::uint64_t*>(
            sycl::malloc_host(cap * sizeof(std::uint64_t), q));
        buf.h_meta        = static_cast<std::uint64_t*>(
            sycl::malloc_host(cap * sizeof(std::uint64_t), q));
        buf.h_t2_meta     = static_cast<std::uint64_t*>(
            sycl::malloc_host(cap * sizeof(std::uint64_t), q));
        buf.h_t2_xbits    = static_cast<std::uint32_t*>(
            sycl::malloc_host(cap * sizeof(std::uint32_t), q));
        buf.h_keys_merged = static_cast<std::uint32_t*>(
            sycl::malloc_host(cap * sizeof(std::uint32_t), q));
        if (!buf.pinned_dst || !buf.h_meta || !buf.h_t2_meta ||
            !buf.h_t2_xbits || !buf.h_keys_merged)
        {
            free_boundary(buf);
            throw std::runtime_error(
                "run_pipeline_parallel_split: boundary pinned-host alloc failed");
        }
    }

    PipelineParallelSplitResult result;
    std::exception_ptr first_exc;
    std::exception_ptr second_exc;

    // First half on device_first.
    std::uint64_t t1_count_out = 0;
    std::uint64_t t2_count_out = 0;
    {
        std::thread t([&] {
            try {
                bind_current_device(device_first);
                StreamingPinnedScratch scratch{};
                scratch.tiny_mode          = (tier_first == PipelineStageTier::Tiny);
                scratch.t2_tile_count      = 8;
                scratch.gather_tile_count  = 4;
                scratch.h_meta             = buf.h_meta;
                scratch.h_t2_meta          = buf.h_t2_meta;
                scratch.h_t2_xbits         = buf.h_t2_xbits;
                scratch.h_keys_merged      = buf.h_keys_merged;
                scratch.stop_after_t2_sort = true;
                auto r = run_gpu_pipeline_streaming(
                    cfg, buf.pinned_dst, cap, scratch);
                t1_count_out = r.t1_count;
                t2_count_out = r.t2_count;
            } catch (...) {
                first_exc = std::current_exception();
            }
        });
        t.join();
    }
    if (first_exc) {
        free_boundary(buf);
        std::rethrow_exception(first_exc);
    }

    // Second half on device_second. The host-pinned boundary buffers
    // are accessible from any CUDA context in the process, so the
    // second device reads + writes them directly.
    {
        std::thread t([&] {
            try {
                bind_current_device(device_second);
                StreamingPinnedScratch scratch{};
                scratch.tiny_mode          = (tier_second == PipelineStageTier::Tiny);
                scratch.t2_tile_count     = 8;
                scratch.gather_tile_count = 4;
                scratch.h_meta            = buf.h_meta;
                scratch.h_t2_meta         = buf.h_t2_meta;
                scratch.h_t2_xbits        = buf.h_t2_xbits;
                scratch.h_keys_merged     = buf.h_keys_merged;
                scratch.start_at_t3_match = true;
                scratch.t1_count_in       = t1_count_out;
                scratch.t2_count_in       = t2_count_out;
                auto r = run_gpu_pipeline_streaming(
                    cfg, buf.pinned_dst, cap, scratch);
                auto frags = r.fragments();
                result.fragments_storage.assign(frags.begin(), frags.end());
                result.t1_count = r.t1_count;
                result.t2_count = r.t2_count;
                result.t3_count = r.t3_count;
            } catch (...) {
                second_exc = std::current_exception();
            }
        });
        t.join();
    }
    if (second_exc) {
        free_boundary(buf);
        std::rethrow_exception(second_exc);
    }

    free_boundary(buf);
    return result;
}

namespace {

// Bounded slot channel for handoff between stage 1 and stage 2.
// stage1 sends `slot_index` after writing into bufs[slot]. stage2
// receives slot_index and reads from bufs[slot]. After stage2
// finishes with bufs[slot], it releases the slot back via
// free_slot.send(slot).
class SlotChannel {
public:
    void send(int slot)
    {
        std::lock_guard<std::mutex> lk(m_);
        q_.push(slot);
        cv_.notify_one();
    }
    void close()
    {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        cv_.notify_all();
    }
    // Returns -1 when channel is closed and empty.
    int recv()
    {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return closed_ || !q_.empty(); });
        if (q_.empty()) return -1;
        int s = q_.front();
        q_.pop();
        return s;
    }

private:
    std::mutex              m_;
    std::condition_variable cv_;
    std::queue<int>         q_;
    bool                    closed_ = false;
};

} // namespace

std::vector<PipelineParallelSplitResult> run_pipeline_parallel_batch(
    std::vector<GpuPipelineConfig> const& cfgs,
    std::vector<int> const&               device_ids,
    int                                   depth,
    std::vector<PipelineStageTier> const& tiers)
{
    if (cfgs.empty()) return {};
    if (depth < 1) depth = 1;
    if (depth > static_cast<int>(cfgs.size())) {
        depth = static_cast<int>(cfgs.size());
    }
    if (device_ids.size() != 2) {
        throw std::runtime_error(
            "run_pipeline_parallel_batch: exactly 2 device ids required (got " +
            std::to_string(device_ids.size()) + "); N-stage in progress");
    }
    if (!tiers.empty() && tiers.size() != device_ids.size()) {
        throw std::runtime_error(
            "run_pipeline_parallel_batch: tiers must be empty or match "
            "device_ids size");
    }
    int const device_first  = device_ids[0];
    int const device_second = device_ids[1];
    PipelineStageTier const tier_first =
        tiers.empty() ? PipelineStageTier::Tiny : tiers[0];
    PipelineStageTier const tier_second =
        tiers.empty() ? PipelineStageTier::Tiny : tiers[1];
    if (device_first < 0 || device_second < 0) {
        throw std::runtime_error(
            "run_pipeline_parallel_batch: device ids must be non-negative");
    }

    // Validate cfgs are uniform on cap-relevant fields. Each plot
    // re-uses the same boundary slots; cap must be the same.
    int const k0 = cfgs[0].k;
    for (auto const& c : cfgs) {
        if (c.k != k0) {
            throw std::runtime_error(
                "run_pipeline_parallel_batch: heterogeneous k across "
                "entries is not supported (would require per-slot caps)");
        }
        if (c.k < 18 || c.k > 32 || (c.k & 1) != 0) {
            throw std::runtime_error("k must be even in [18, 32]");
        }
        if (c.strength < 2) {
            throw std::runtime_error("strength must be >= 2");
        }
    }

    int const num_section_bits = (k0 < 28) ? 2 : (k0 - 26);
    std::uint64_t const cap =
        max_pairs_per_section(k0, num_section_bits) *
        (std::uint64_t{1} << num_section_bits);

    // Allocate `depth` boundary buffer sets on dev_first's queue.
    bind_current_device(device_first);
    sycl::queue& alloc_q = sycl_backend::queue();
    std::vector<BoundaryBuffers> bufs(depth);
    for (int i = 0; i < depth; ++i) {
        bufs[i].alloc_queue = &alloc_q;
        bufs[i].pinned_dst    = static_cast<std::uint64_t*>(
            sycl::malloc_host(cap * sizeof(std::uint64_t), alloc_q));
        bufs[i].h_meta        = static_cast<std::uint64_t*>(
            sycl::malloc_host(cap * sizeof(std::uint64_t), alloc_q));
        bufs[i].h_t2_meta     = static_cast<std::uint64_t*>(
            sycl::malloc_host(cap * sizeof(std::uint64_t), alloc_q));
        bufs[i].h_t2_xbits    = static_cast<std::uint32_t*>(
            sycl::malloc_host(cap * sizeof(std::uint32_t), alloc_q));
        bufs[i].h_keys_merged = static_cast<std::uint32_t*>(
            sycl::malloc_host(cap * sizeof(std::uint32_t), alloc_q));
        if (!bufs[i].pinned_dst || !bufs[i].h_meta || !bufs[i].h_t2_meta ||
            !bufs[i].h_t2_xbits || !bufs[i].h_keys_merged)
        {
            for (auto& b : bufs) free_boundary(b);
            throw std::runtime_error(
                "run_pipeline_parallel_batch: boundary alloc failed");
        }
    }

    // Per-slot state shared between stage 1 and stage 2.
    struct SlotState {
        std::uint64_t t1_count = 0;
        std::uint64_t t2_count = 0;
        int           cfg_idx  = -1;
    };
    std::vector<SlotState> slot_state(depth);

    // free_slots: slots ready for stage 1 to fill. Pre-fill with all.
    // ready_slots: slots that stage 1 has filled, awaiting stage 2.
    SlotChannel free_slots;
    SlotChannel ready_slots;
    for (int i = 0; i < depth; ++i) free_slots.send(i);

    std::vector<PipelineParallelSplitResult> results(cfgs.size());
    std::exception_ptr stage1_exc;
    std::exception_ptr stage2_exc;

    std::thread stage1([&] {
        try {
            bind_current_device(device_first);
            // Per-thread host-pinned pool: amortises per-plot allocs
            // (h_t1_mi, h_t2_mi) across all plots this thread handles.
            HostPinnedPool stage1_pool;
            for (std::size_t idx = 0; idx < cfgs.size(); ++idx) {
                int const slot = free_slots.recv();
                if (slot < 0) break;
                StreamingPinnedScratch s{};
                s.tiny_mode          = (tier_first == PipelineStageTier::Tiny);
                s.t2_tile_count      = 8;
                s.gather_tile_count  = 4;
                s.h_meta             = bufs[slot].h_meta;
                s.h_t2_meta          = bufs[slot].h_t2_meta;
                s.h_t2_xbits         = bufs[slot].h_t2_xbits;
                s.h_keys_merged      = bufs[slot].h_keys_merged;
                s.stop_after_t2_sort = true;
                s.pool               = &stage1_pool;
                auto r = run_gpu_pipeline_streaming(
                    cfgs[idx], bufs[slot].pinned_dst, cap, s);
                slot_state[slot].t1_count = r.t1_count;
                slot_state[slot].t2_count = r.t2_count;
                slot_state[slot].cfg_idx  = static_cast<int>(idx);
                ready_slots.send(slot);
            }
        } catch (...) {
            stage1_exc = std::current_exception();
        }
        ready_slots.close();
    });

    std::thread stage2([&] {
        try {
            bind_current_device(device_second);
            // Per-thread host-pinned pool: amortises h_t3 across plots.
            HostPinnedPool stage2_pool;
            for (;;) {
                int const slot = ready_slots.recv();
                if (slot < 0) break;
                int const idx = slot_state[slot].cfg_idx;
                StreamingPinnedScratch s{};
                s.tiny_mode          = (tier_second == PipelineStageTier::Tiny);
                s.t2_tile_count     = 8;
                s.gather_tile_count = 4;
                s.h_meta            = bufs[slot].h_meta;
                s.h_t2_meta         = bufs[slot].h_t2_meta;
                s.h_t2_xbits        = bufs[slot].h_t2_xbits;
                s.h_keys_merged     = bufs[slot].h_keys_merged;
                s.start_at_t3_match = true;
                s.t1_count_in       = slot_state[slot].t1_count;
                s.t2_count_in       = slot_state[slot].t2_count;
                s.pool              = &stage2_pool;
                auto r = run_gpu_pipeline_streaming(
                    cfgs[idx], bufs[slot].pinned_dst, cap, s);
                auto frags = r.fragments();
                results[idx].fragments_storage.assign(
                    frags.begin(), frags.end());
                results[idx].t1_count = r.t1_count;
                results[idx].t2_count = r.t2_count;
                results[idx].t3_count = r.t3_count;
                free_slots.send(slot);
            }
        } catch (...) {
            stage2_exc = std::current_exception();
        }
    });

    stage1.join();
    stage2.join();

    for (auto& b : bufs) free_boundary(b);

    if (stage1_exc) std::rethrow_exception(stage1_exc);
    if (stage2_exc) std::rethrow_exception(stage2_exc);
    return results;
}

} // namespace pos2gpu
