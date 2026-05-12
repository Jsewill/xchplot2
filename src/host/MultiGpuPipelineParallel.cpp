// MultiGpuPipelineParallel.cpp — Phase 2.1c/d orchestrator. See header.

#include "host/MultiGpuPipelineParallel.hpp"

#include "gpu/SyclBackend.hpp"
#include "host/HostPinnedPool.hpp"
#include "host/PoolSizing.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
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

namespace {

// Heaviness order for N stages — the stage at index 0 of the returned
// vector is the heaviest (gets the largest-VRAM device); index N-1 is
// the lightest. Estimates without per-stage VRAM-peak measurement:
//   N=2: stage 0 (Xs+T1+T2) > stage 1 (T3+Frag) — Xs candidate stream
//        + CUB radix scratch dominate stage 0 in every tier.
//   N=3: stage 1 (T2 match+sort) > stage 0 (Xs+T1) > stage 2 (T3+Frag)
//        — T2 holds cap-sized device output AND CUB sort scratch
//        concurrently; Xs+T1 alone is lighter; T3+Frag is the
//        smallest.
// Update once per-stage VRAM peaks are measured.
std::vector<int> stage_heaviness_order(int N)
{
    if (N == 2) return {0, 1};
    if (N == 3) return {1, 0, 2};
    throw std::runtime_error(
        "select_pipeline_devices: heaviness order undefined for N=" +
        std::to_string(N) + " (only 2 and 3 are implemented)");
}

} // namespace

PipelineDeviceAssignment select_pipeline_devices(
    std::vector<int> const&                  device_ids,
    std::function<std::uint64_t(int)> const& vram_for_device)
{
    int const N = static_cast<int>(device_ids.size());
    if (N != 2 && N != 3) {
        throw std::runtime_error(
            "select_pipeline_devices: device_ids.size() must be 2 or 3 (got " +
            std::to_string(N) + ")");
    }
    for (int id : device_ids) {
        if (id < 0) throw std::runtime_error(
            "select_pipeline_devices: device ids must be non-negative");
    }

    auto vram_lookup = [&](int idx) { return vram_for_device(device_ids[idx]); };

    PipelineDeviceAssignment out;
    out.dev_ids.assign(N, -1);
    out.dev_vram_bytes.assign(N, 0);

    // Uniform-VRAM rig: keep caller's order (no reason to shuffle
    // identical cards, and the user's slot-order preference may matter).
    // Matches the 2-stage tie-keeps-order behaviour.
    bool all_vram_equal = true;
    auto const v0 = vram_lookup(0);
    for (int i = 1; i < N; ++i) {
        if (vram_lookup(i) != v0) { all_vram_equal = false; break; }
    }
    if (all_vram_equal) {
        for (int i = 0; i < N; ++i) {
            out.dev_ids[i]        = device_ids[i];
            out.dev_vram_bytes[i] = v0;
        }
        out.reordered = false;
    } else {
        // Heterogeneous: sort device indices VRAM-descending and
        // assign biggest-VRAM device to heaviest stage.
        std::vector<int> vram_rank(N);
        for (int i = 0; i < N; ++i) vram_rank[i] = i;
        std::stable_sort(vram_rank.begin(), vram_rank.end(),
                         [&](int a, int b) { return vram_lookup(a) > vram_lookup(b); });

        std::vector<int> const heaviness = stage_heaviness_order(N);
        for (int r = 0; r < N; ++r) {
            int const stage = heaviness[r];
            int const src   = vram_rank[r];
            out.dev_ids[stage]        = device_ids[src];
            out.dev_vram_bytes[stage] = vram_lookup(src);
        }
        for (int i = 0; i < N; ++i) {
            if (out.dev_ids[i] != device_ids[i]) { out.reordered = true; break; }
        }
    }

    // Backward-compat scalar fields for N=2 callers + existing parity
    // tests. Left default-zero for N>2; new code reads dev_ids[].
    if (N == 2) {
        out.dev_first             = out.dev_ids[0];
        out.dev_second            = out.dev_ids[1];
        out.dev_first_vram_bytes  = out.dev_vram_bytes[0];
        out.dev_second_vram_bytes = out.dev_vram_bytes[1];
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
    int const N = static_cast<int>(device_ids.size());
    if (N != 2 && N != 3) {
        throw std::runtime_error(
            "run_pipeline_parallel_batch: device_ids.size() must be 2 (T2-sort "
            "split) or 3 (T1-sort + T2-sort split); got " + std::to_string(N));
    }
    if (!tiers.empty() && tiers.size() != device_ids.size()) {
        throw std::runtime_error(
            "run_pipeline_parallel_batch: tiers must be empty or match "
            "device_ids size");
    }
    for (int id : device_ids) {
        if (id < 0) throw std::runtime_error(
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

    // Per-stage role (which streaming-pipeline phase boundary to use).
    // N=2 (one boundary, T2-sort): stage 0 = Xs+T1+T2, stage 1 = T3+Frag.
    // N=3 (two boundaries): stage 0 = Xs+T1, stage 1 = T2 match+sort,
    //                       stage 2 = T3+Frag.
    struct StageRole {
        int               device_id          = -1;
        std::uint64_t     vram_bytes         = 0;
        PipelineStageTier tier               = PipelineStageTier::Tiny;
        bool              stop_after_t1_sort = false;
        bool              stop_after_t2_sort = false;
        bool              start_at_t2_match  = false;
        bool              start_at_t3_match  = false;
        bool              t3_sort_full_cap   = false;
    };
    std::vector<StageRole> roles(N);
    // Look up each staged device's VRAM up-front. Used to gate the
    // Phase 2.5a t3_sort_full_cap optimisation on the final stage.
    auto const& sycl_devs = sycl_backend::usable_gpu_devices();
    for (int i = 0; i < N; ++i) {
        roles[i].device_id = device_ids[i];
        roles[i].tier      = tiers.empty() ? PipelineStageTier::Tiny : tiers[i];
        roles[i].vram_bytes =
            (device_ids[i] >= 0 &&
             static_cast<std::size_t>(device_ids[i]) < sycl_devs.size())
            ? sycl_devs[device_ids[i]].get_info<sycl::info::device::global_mem_size>()
            : 0;
    }
    if (N == 2) {
        roles[0].stop_after_t2_sort = true;
        roles[1].start_at_t3_match  = true;
    } else { // N == 3
        roles[0].stop_after_t1_sort = true;
        roles[1].start_at_t2_match  = true;
        roles[1].stop_after_t2_sort = true;
        roles[2].start_at_t3_match  = true;
    }

    // Phase 2.5a: enable the full-cap on-device T3 sort for the final
    // stage when its device has VRAM headroom. The minimal-tier tile-
    // merge path (host std::inplace_merge) bloats T3 sort wall by 15-
    // 30× in pipelined-batch mode under PCIe contention. Threshold is
    // 6 GB (full-cap T3 sort peaks at ~4.2 GB at k=28; +40% safety).
    // Tiny stages can't use this — their input is host-pinned, no
    // device-resident d_t3.
    constexpr std::uint64_t kT3FullCapVramFloor = 6ULL << 30; // 6 GB
    int const final_stage = N - 1;
    if (roles[final_stage].tier == PipelineStageTier::Minimal &&
        roles[final_stage].vram_bytes >= kT3FullCapVramFloor)
    {
        roles[final_stage].t3_sort_full_cap = true;
    }

    // Allocate `depth` boundary buffer sets on stage-0's queue. The
    // single BoundaryBuffers shape covers both T1-sort and T2-sort
    // boundaries — T1-sort uses h_meta + h_keys_merged only, T2-sort
    // uses all four host pinned buffers; pinned_dst is the final-stage
    // output. Each slot is exclusively owned by one plot at a time as
    // it flows through the stages, so reusing buffers across boundaries
    // within a slot is safe.
    bind_current_device(roles[0].device_id);
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

    // Per-slot counts that downstream stages need (echoed forward).
    struct SlotState {
        std::uint64_t t1_count = 0;
        std::uint64_t t2_count = 0;
        int           cfg_idx  = -1;
    };
    std::vector<SlotState> slot_state(depth);

    // Channel topology: channels[0] = free slots (recycled by the last
    // stage). channels[i] for 1 <= i <= N-1 = ready slots between stage
    // i-1 and stage i. SlotChannel has std::mutex so it's non-movable;
    // store via unique_ptr to keep references stable.
    std::vector<std::unique_ptr<SlotChannel>> channels(N);
    for (int i = 0; i < N; ++i) channels[i] = std::make_unique<SlotChannel>();
    for (int i = 0; i < depth; ++i) channels[0]->send(i);

    std::vector<PipelineParallelSplitResult> results(cfgs.size());
    std::vector<std::exception_ptr> excs(N);

    std::vector<std::thread> stage_threads;
    stage_threads.reserve(N);
    for (int s = 0; s < N; ++s) {
        stage_threads.emplace_back([&, s] {
            try {
                bind_current_device(roles[s].device_id);
                // Per-thread host-pinned pool: amortises per-plot allocs
                // (h_t1_mi, h_t2_mi, h_t3, h_keys_merged, h_merged_vals)
                // across all plots this thread handles.
                HostPinnedPool stage_pool;

                bool const is_first = (s == 0);
                bool const is_last  = (s == N - 1);
                std::size_t plot_idx = 0;

                for (;;) {
                    int const slot = channels[s]->recv();
                    if (slot < 0) break;

                    int cfg_idx;
                    if (is_first) {
                        if (plot_idx >= cfgs.size()) break;
                        cfg_idx = static_cast<int>(plot_idx++);
                    } else {
                        cfg_idx = slot_state[slot].cfg_idx;
                    }

                    StreamingPinnedScratch sc{};
                    sc.tiny_mode          = (roles[s].tier == PipelineStageTier::Tiny);
                    sc.t2_tile_count      = 8;
                    sc.gather_tile_count  = 4;
                    sc.h_meta             = bufs[slot].h_meta;
                    sc.h_t2_meta          = bufs[slot].h_t2_meta;
                    sc.h_t2_xbits         = bufs[slot].h_t2_xbits;
                    sc.h_keys_merged      = bufs[slot].h_keys_merged;
                    sc.stop_after_t1_sort = roles[s].stop_after_t1_sort;
                    sc.stop_after_t2_sort = roles[s].stop_after_t2_sort;
                    sc.start_at_t2_match  = roles[s].start_at_t2_match;
                    sc.start_at_t3_match  = roles[s].start_at_t3_match;
                    sc.t3_sort_full_cap   = roles[s].t3_sort_full_cap;
                    sc.t1_count_in        = slot_state[slot].t1_count;
                    sc.t2_count_in        = slot_state[slot].t2_count;
                    sc.pool               = &stage_pool;

                    auto r = run_gpu_pipeline_streaming(
                        cfgs[cfg_idx], bufs[slot].pinned_dst, cap, sc);

                    if (is_first) {
                        slot_state[slot].cfg_idx = cfg_idx;
                    }
                    if (!is_last) {
                        // Echo forward — the streaming pipeline copies
                        // input counts back into result when relevant
                        // (start_at_*) and writes new counts otherwise.
                        slot_state[slot].t1_count = r.t1_count;
                        slot_state[slot].t2_count = r.t2_count;
                        channels[s + 1]->send(slot);
                    } else {
                        // Final stage: capture fragments + recycle slot.
                        auto frags = r.fragments();
                        results[cfg_idx].fragments_storage.assign(
                            frags.begin(), frags.end());
                        results[cfg_idx].t1_count = r.t1_count;
                        results[cfg_idx].t2_count = r.t2_count;
                        results[cfg_idx].t3_count = r.t3_count;
                        channels[0]->send(slot);
                    }
                }
            } catch (...) {
                excs[s] = std::current_exception();
            }
            // Close downstream so the next stage can drain and exit.
            // Last stage doesn't close channels[0] — channels[0] just
            // becomes unreferenced when its thread exits.
            if (s + 1 < N) {
                channels[s + 1]->close();
            }
        });
    }

    for (auto& t : stage_threads) t.join();

    for (auto& b : bufs) free_boundary(b);

    for (auto& e : excs) if (e) std::rethrow_exception(e);
    return results;
}

} // namespace pos2gpu
