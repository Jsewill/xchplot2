#include "host/GroupGpu.hpp"
#include "host/GroupGpuPacked.hpp"

#include "gpu/SyclBackend.hpp"
#include "host/GpuBufferPool.hpp"
#include "host/GpuPipeline.hpp"
#include "host/HostGuard.hpp"
#include "host/HostPinnedPool.hpp"
#include "host/PoolSizing.hpp"
#include "host/VramBudget.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace group_grow
{
namespace
{
// Same fixed runtime/process allowance as GpuBufferPool.cpp's host estimates.
// Pinned allocations and the owning output are accounted for separately below.
constexpr uint64_t runtime_host_bytes = uint64_t(1300) << 20;
// Match the production spill engine's aggregate pinned staging allocation.
// The device drain splits it into alternating windows without growing the budget.
constexpr uint64_t transfer_window_bytes = uint64_t(64) << 20;

uint64_t wall_stamp()
{
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct AddWallTime
{
    uint64_t& total;
    uint64_t begin;
    AddWallTime(uint64_t& total_, bool enabled) : total(total_), begin(enabled ? wall_stamp() : 0) {}
    ~AddWallTime() { if (begin) total += wall_stamp() - begin; }
};

// These are host wall intervals, sharing steady_clock's epoch with controller
// diagnostics. Copy waits are residual blocking after overlap, not copy engine
// execution time. Pipeline phase timing and an external device timeline remain
// necessary to distinguish kernel execution from runtime/host work.
struct OutputWallTimings
{
    bool enabled, complete = false, observes_copy_wait = false;
    char const* event;
    uint64_t member, begin;
    uint64_t submit_ns = 0, transfer_wait_ns = 0, consume_ns = 0, slot_wait_ns = 0, bytes = 0;
    uint64_t device_copy_ns = 0, profiled_copies = 0, copy_profile_failures = 0;

    OutputWallTimings(bool enabled_, char const* event_, uint64_t member_)
        : enabled(enabled_), event(event_), member(member_), begin(enabled ? wall_stamp() : 0) {}
    ~OutputWallTimings()
    {
        if (!enabled) return;
        uint64_t const end = wall_stamp();
        std::fprintf(stderr, "[grow-gpu-timing] event=%s member=%llu begin_ns=%llu end_ns=%llu "
            "status=%s copy_wait_observed=%u submit_wall_ns=%llu transfer_wait_wall_ns=%llu "
            "consume_wall_ns=%llu output_slot_wait_wall_ns=%llu transfer_bytes=%llu "
            "device_copy_ns=%llu profiled_copies=%llu copy_profile_failures=%llu\n",
            event, static_cast<unsigned long long>(member), static_cast<unsigned long long>(begin),
            static_cast<unsigned long long>(end), complete ? "complete" : "failed", unsigned(observes_copy_wait),
            static_cast<unsigned long long>(submit_ns), static_cast<unsigned long long>(transfer_wait_ns),
            static_cast<unsigned long long>(consume_ns), static_cast<unsigned long long>(slot_wait_ns),
            static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(device_copy_ns),
            static_cast<unsigned long long>(profiled_copies), static_cast<unsigned long long>(copy_profile_failures));
    }
};

struct TimedCopyEvent
{
    sycl::event event;
    OutputWallTimings& wall;
    bool profile_event;
    void wait_and_throw()
    {
        {
            AddWallTime const timing(wall.transfer_wait_ns, wall.enabled);
            event.wait_and_throw();
        }
        if (profile_event)
        {
            try
            {
                auto const start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto const end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
                if (end >= start)
                {
                    wall.device_copy_ns += end - start;
                    ++wall.profiled_copies;
                }
                else ++wall.copy_profile_failures;
            }
            catch (sycl::exception const&) { ++wall.copy_profile_failures; }
        }
    }
};

struct FreeU64
{
    void operator()(uint64_t* p) const { if (p) pos2gpu::streaming_free_pinned_uint64(p); }
};
struct FreeU32
{
    void operator()(uint32_t* p) const { if (p) pos2gpu::streaming_free_pinned_uint32(p); }
};
using PinnedU64 = std::unique_ptr<uint64_t, FreeU64>;
using PinnedU32 = std::unique_ptr<uint32_t, FreeU32>;

uint64_t device_peak(int k, std::string const& tier)
{
    if (tier == "plain") return pos2gpu::streaming_plain_peak_bytes(k);
    if (tier == "compact") return pos2gpu::streaming_peak_bytes(k);
    if (tier == "minimal") return pos2gpu::streaming_minimal_peak_bytes(k);
    if (tier == "tiny") return pos2gpu::streaming_tiny_peak_bytes(k);
    return pos2gpu::streaming_pinned_peak_bytes(k);
}

uint64_t host_peak(int k, std::string const& tier)
{
    if (tier == "pool" || tier == "plain") return pos2gpu::streaming_plain_host_bytes(k);
    if (tier == "compact") return pos2gpu::streaming_compact_host_bytes(k);
    if (tier == "minimal") return pos2gpu::streaming_minimal_host_bytes(k);
    return pos2gpu::streaming_tiny_host_bytes(k);
}
} // namespace

struct GpuProducer::Impl
{
    int k, strength;
    GpuProducerOptions options;
    uint64_t capacity;
    uint64_t twophase_budget_bytes = 0;
    bool profile = false;
    bool copy_event_profiling = false;
    uint64_t profile_member = 0;
    sycl::queue* queue = nullptr;
    detail::OutputHandoff handoff;
    std::string selected;
    std::unique_ptr<pos2gpu::GpuBufferPool> pool;
    std::unique_ptr<sycl::queue> copy_queue;
    uint64_t* device_output = nullptr;
    uint64_t* device_packed = nullptr;
    uint64_t packed_scratch_words = 0;
    pos2gpu::HostPinnedPool host_pool;
    PinnedU64 output, transfer, meta, t3, t2_meta;
    PinnedU32 keys, xbits;

    Impl(int k_, int strength_, GpuProducerOptions options_)
        : k(k_), strength(strength_), options(std::move(options_)), capacity(0)
    {
        char const* const phase_timing = std::getenv("POS2GPU_PHASE_TIMING");
        profile = options.profile || (phase_timing && phase_timing[0] == '1');
        if (k < 18 || k > 28 || (k & 1) || strength < 2 || strength > k - 3)
            throw std::invalid_argument("GPU producer requires even k in [18, 28] and strength in [2, k-3]");
        if (options.tier != "auto" && options.tier != "pool" && options.tier != "plain" &&
            options.tier != "compact" && options.tier != "minimal" &&
            options.tier != "tiny" && options.tier != "pinned")
            throw std::invalid_argument("GPU tier must be auto, pool, plain, compact, minimal, tiny, or pinned");
        if (pos2gpu::host_guard_on())
            throw std::invalid_argument("GPU growth RAM admission requires XCHPLOT2_HOST_GUARD to be disabled");
        if (options.device_id < 0 || options.device_id >= pos2gpu::gpu_device_count())
            throw std::invalid_argument("GPU device index is outside the available GPU inventory");
        if (char const* v = std::getenv("XCHPLOT2_SPILL_T1META"); v && v[0] == '1')
            throw std::invalid_argument("same-file GPU growth does not permit XCHPLOT2_SPILL_T1META disk offload");
        pos2gpu::bind_current_device(options.device_id);
        capacity = pos2gpu::match_phase_capacity(k, k < 28 ? 2 : k - 26);
        queue = &pos2gpu::sycl_backend::queue();
        // Match production admission: discard earlier cached scratch before
        // sampling free VRAM, and reserve the complete path peak before grants.
        pos2gpu::sycl_backend::set_twophase_budget(*queue, 0);
        auto memory_before_pool = pos2gpu::query_device_memory();
        if (options.max_vram_bytes)
            memory_before_pool.free_bytes = std::min<uint64_t>(memory_before_pool.free_bytes, options.max_vram_bytes);
        auto const margin = pos2gpu::vram_safety_margin();

        if (options.tier == "auto" || options.tier == "pool")
        {
            try
            {
                pool = std::make_unique<pos2gpu::GpuBufferPool>(k, strength, false, options.max_vram_bytes);
                // pair_a is lazy. Free VRAM after construction cannot safely
                // grant scratch without also reserving that full allocation.
                auto const required = pool->storage_bytes + pool->pair_a_bytes +
                    pool->pair_b_bytes + pool->sort_scratch_bytes + sizeof(uint64_t);
                twophase_budget_bytes = pos2gpu::vram_scratch_budget(
                    memory_before_pool.free_bytes, required, margin);
                auto const overlap_budget = pos2gpu::vram_scratch_budget(
                    memory_before_pool.free_bytes, required + capacity * sizeof(uint64_t), margin);
                auto const buckets = uint64_t(1) << ((k < 28 ? 2 : k - 26) + strength);
                auto const match_scratch = pos2gpu::twophase_scratch_bytes(capacity, buckets);
                if (options.member_overlap && pos2gpu::vram_fits(memory_before_pool.free_bytes,
                        required + capacity * sizeof(uint64_t), margin) &&
                    (!options.preserve_match_scratch || detail::preserves_match_scratch(
                        twophase_budget_bytes, overlap_budget, match_scratch)))
                {
                    // This optional allocation must not turn an otherwise
                    // admitted pooled run into CPU/streaming fallback when a
                    // concurrent GPU tenant takes the extra headroom.
                    try
                    {
                        // Construct the optional queue before allocating its
                        // slot, so queue failure has no device allocation to
                        // unwind. Both failures leave the serial pool usable.
                        std::unique_ptr<sycl::queue> candidate_queue;
                        if (profile)
                        {
                            try
                            {
                                candidate_queue = std::make_unique<sycl::queue>(queue->get_context(),
                                    queue->get_device(), pos2gpu::sycl_backend::async_error_handler,
                                    sycl::property_list{sycl::property::queue::in_order{},
                                                        sycl::property::queue::enable_profiling{}});
                                copy_event_profiling = true;
                            }
                            catch (sycl::exception const&) {}
                        }
                        if (!candidate_queue)
                            candidate_queue = std::make_unique<sycl::queue>(queue->get_context(),
                                queue->get_device(), pos2gpu::sycl_backend::async_error_handler,
                                sycl::property::queue::in_order{});
                        auto* candidate_output = sycl::malloc_device<uint64_t>(capacity, *queue);
                        if (candidate_output)
                        {
                            device_output = candidate_output;
                            copy_queue = std::move(candidate_queue);
                            // Reserve the slot and the lazy pair_a together,
                            // then grant only the remaining match scratch.
                            twophase_budget_bytes = overlap_budget;
                            // Packed transport is optional even when raw
                            // member overlap fits. Reserve its complete tile
                            // beside the lazy pool and independent output;
                            // preserve an already admitted fast match path.
                            if (options.compact_transfer && capacity <= UINT32_MAX)
                            {
                                detail::PackedTransferLayout const layout(k, capacity);
                                uint64_t const words = (transfer_capacity() + 1) / 2;
                                uint64_t const packed_peak = required + capacity * sizeof(uint64_t) + words * sizeof(uint64_t);
                                uint64_t const packed_budget = pos2gpu::vram_scratch_budget(
                                    memory_before_pool.free_bytes, packed_peak, margin);
                                if (words && layout.metadata_words <= transfer_capacity() / 2 &&
                                    pos2gpu::vram_fits(memory_before_pool.free_bytes, packed_peak, margin) &&
                                    (!options.preserve_match_scratch || detail::preserves_match_scratch(
                                        overlap_budget, packed_budget, match_scratch)))
                                {
                                    try
                                    {
                                        device_packed = sycl::malloc_device<uint64_t>(words, *queue);
                                        if (device_packed)
                                        {
                                            packed_scratch_words = words;
                                            twophase_budget_bytes = packed_budget;
                                        }
                                    }
                                    catch (std::exception const&) {}
                                }
                            }
                        }
                    }
                    catch (std::exception const&) {}
                }
                selected = "pool";
                return;
            }
            catch (pos2gpu::InsufficientVramError const&)
            {
                if (options.tier == "pool") throw;
            }
        }
        auto memory = pos2gpu::query_device_memory();
        if (options.max_vram_bytes)
            memory.free_bytes = std::min<uint64_t>(memory.free_bytes, options.max_vram_bytes);
        selected = options.tier;
        if (selected == "auto")
        {
            for (char const* candidate : {"plain", "compact", "minimal", "tiny"})
            {
                selected = candidate;
                if (pos2gpu::vram_fits(memory.free_bytes, device_peak(k, selected), margin)) break;
            }
        }
        auto const required = device_peak(k, selected);
        if (!pos2gpu::vram_fits(memory.free_bytes, required, margin))
            throw pos2gpu::InsufficientVramError("GPU growth tier peak plus VRAM buffer exceeds free device memory");
        twophase_budget_bytes = pos2gpu::vram_scratch_budget(memory.free_bytes, required, margin);
    }

    ~Impl()
    {
        // The queue belongs to the worker thread and outlives this adapter.
        // Match production queue cleanup: wait for users, release the shared
        // device cache, and report cleanup failure without throwing on unwind.
        if (!queue) return;
        auto cleanup = [](char const* name, auto&& action)
        {
            try { action(); }
            catch (std::exception const& error)
            { std::fprintf(stderr, "%s cleanup failed: %s\n", name, error.what()); }
            catch (...)
            { std::fprintf(stderr, "%s cleanup failed: unknown exception\n", name); }
        };
        cleanup("GPU output", [&] { handoff.finish(); });
        if (device_output)
        {
            // A completed wait can still report an asynchronous error. Attempt
            // every remaining wait/free independently instead of leaking all
            // later resources after the first error.
            cleanup("GPU generation queue", [&] { queue->wait_and_throw(); });
            cleanup("GPU copy queue", [&] { copy_queue->wait_and_throw(); });
            cleanup("GPU device output", [&]
            {
                sycl::free(device_output, *queue);
                device_output = nullptr;
            });
            cleanup("GPU compact output", [&]
            {
                if (device_packed) sycl::free(device_packed, *queue);
                device_packed = nullptr;
            });
        }
        cleanup("two-phase scratch", [&] { pos2gpu::sycl_backend::release_twophase_scratch(*queue); });
    }

    void check_generation() const
    {
        handoff.check_generation();
    }

    uint64_t next_profile_member() { return profile ? ++profile_member : 0; }

    uint64_t finish_output(uint64_t member = 0)
    {
        if (!profile || !handoff.pending()) { handoff.finish(); return 0; }
        OutputWallTimings wall(true, "output-slot-wait", member);
        handoff.finish();
        wall.complete = true;
        return wall_stamp() - wall.begin;
    }

    uint64_t transfer_capacity() const
    {
        return std::min(capacity, transfer_window_bytes / sizeof(uint64_t));
    }

    void prepare_tiled_output()
    {
        if (pool) pool->release_pinned(0);
        else release(output);
        if (!transfer)
            transfer.reset(pos2gpu::streaming_alloc_pinned_uint64(transfer_capacity()));
        if (!transfer) throw std::runtime_error("GPU growth pinned transfer allocation failed");
    }

    pos2gpu::GpuPipelineConfig config(std::array<uint8_t, 32> const& plot_id) const
    {
        pos2gpu::GpuPipelineConfig result{};
        result.plot_id = plot_id;
        result.k = k;
        result.strength = strength;
        result.profile = options.profile;
        result.host_threads = options.host_threads;
        result.twophase_budget_bytes = twophase_budget_bytes;
        return result;
    }

    pos2gpu::StreamingPinnedScratch scratch(bool full_output = true)
    {
        if (full_output && !output) output.reset(pos2gpu::streaming_alloc_pinned_uint64(capacity));
        pos2gpu::StreamingPinnedScratch result{};
        result.pool = &host_pool;
        result.twophase_budget_bytes = twophase_budget_bytes;
        result.plain_mode = selected == "plain";
        result.tiny_mode = selected == "tiny" || selected == "pinned";
        result.pinned_mode = selected == "pinned";
        if (selected == "minimal" || result.tiny_mode)
        {
            result.t2_tile_count = 8;
            result.gather_tile_count = 4;
        }
        if (!result.plain_mode)
        {
            if (!meta) meta.reset(pos2gpu::streaming_alloc_pinned_uint64(capacity));
            if (!keys) keys.reset(pos2gpu::streaming_alloc_pinned_uint32(capacity));
            if (!xbits) xbits.reset(pos2gpu::streaming_alloc_pinned_uint32(capacity));
            if (!t3) t3.reset(pos2gpu::streaming_alloc_pinned_uint64(capacity));
            if (result.tiny_mode && !t2_meta)
                t2_meta.reset(pos2gpu::streaming_alloc_pinned_uint64(capacity));
            if (!meta || !keys || !xbits || !t3 || (result.tiny_mode && !t2_meta))
                throw std::runtime_error("GPU growth pinned scratch allocation failed");
            result.h_meta = meta.get();
            result.h_keys_merged = keys.get();
            result.h_t2_xbits = xbits.get();
            result.h_t3 = t3.get();
            result.h_t2_meta = t2_meta.get();
        }
        if (full_output && !output) throw std::runtime_error("GPU growth pinned output allocation failed");
        // All host tables remain in RAM. Device match scratch is bounded by
        // the production grant above; an insufficient grant falls back in-place.
        return result;
    }

    uint64_t output_bytes() const
    {
        if (pool && pool->h_pinned_t3[0]) return pool->pinned_bytes;
        return output ? capacity * sizeof(uint64_t) : 0;
    }

    void release(PinnedU64& buffer)
    {
        if (!buffer) return;
        queue->wait_and_throw();
        pos2gpu::streaming_free_pinned_uint64(buffer.get());
        (void)buffer.release();
    }

    uint64_t retained() const
    {
        uint64_t bytes = runtime_host_bytes + sizeof(Impl) + host_pool.live_bytes();
        if (pool && pool->h_pinned_t3[0]) bytes += pool->pinned_bytes;
        for (bool allocated : {bool(output), bool(meta), bool(t3), bool(t2_meta)})
            if (allocated) bytes += capacity * sizeof(uint64_t);
        for (bool allocated : {bool(keys), bool(xbits)})
            if (allocated) bytes += capacity * sizeof(uint32_t);
        if (transfer) bytes += transfer_capacity() * sizeof(uint64_t);
        return bytes;
    }
};

GpuProducer::GpuProducer(int k, int strength, GpuProducerOptions options)
    : impl_(std::make_unique<Impl>(k, strength, std::move(options))) {}
GpuProducer::~GpuProducer() = default;

GpuProducerPreflight GpuProducer::preflight(int k, int strength, GpuProducerOptions const& options)
{
    if (k < 18 || k > 28 || (k & 1) || strength < 2 || strength > k - 3)
        throw std::invalid_argument("GPU producer requires even k in [18, 28] and strength in [2, k-3]");
    if (options.tier != "auto" && options.tier != "pool" && options.tier != "plain" &&
        options.tier != "compact" && options.tier != "minimal" && options.tier != "tiny" && options.tier != "pinned")
        throw std::invalid_argument("GPU tier must be auto, pool, plain, compact, minimal, tiny, or pinned");
    if (pos2gpu::host_guard_on())
        throw std::invalid_argument("GPU growth RAM admission requires XCHPLOT2_HOST_GUARD to be disabled");
    if (char const* v = std::getenv("XCHPLOT2_SPILL_T1META"); v && v[0] == '1')
        throw std::invalid_argument("same-file GPU growth does not permit XCHPLOT2_SPILL_T1META disk offload");
    pos2gpu::bind_current_device(options.device_id);
    auto memory = pos2gpu::query_device_memory();
    if (options.max_vram_bytes) memory.free_bytes = std::min(memory.free_bytes, options.max_vram_bytes);
    auto const margin = pos2gpu::vram_safety_margin();
    auto const capacity = pos2gpu::match_phase_capacity(k, k < 28 ? 2 : k - 26);
    GpuProducerPreflight result;
    result.tier = options.tier;
    if (result.tier == "auto" || result.tier == "pool")
    {
        auto const sizing = pos2gpu::GpuBufferPool::sizing(k);
        if (pos2gpu::vram_fits(memory.free_bytes, sizing.device_bytes(), margin))
        {
            result.tier = "pool";
            result.device_bytes = sizing.device_bytes() + margin;
            result.borrowed_host_bytes = runtime_host_bytes + sizeof(Impl) + sizing.pinned_bytes;
        }
        else if (result.tier == "pool")
            throw pos2gpu::InsufficientVramError("GPU growth pool exceeds offered device memory");
    }
    if (!result.device_bytes)
    {
        if (result.tier == "auto")
            for (char const* tier : {"plain", "compact", "minimal", "tiny"})
            {
                result.tier = tier;
                if (pos2gpu::vram_fits(memory.free_bytes, device_peak(k, result.tier), margin)) break;
            }
        auto const required = device_peak(k, result.tier);
        if (!pos2gpu::vram_fits(memory.free_bytes, required, margin))
            throw pos2gpu::InsufficientVramError("GPU growth tier exceeds offered device memory");
        result.device_bytes = required + margin;
        result.borrowed_host_bytes = std::max(host_peak(k, result.tier),
            runtime_host_bytes + sizeof(Impl) + capacity * sizeof(uint64_t));
    }
    result.owning_host_bytes = result.borrowed_host_bytes + capacity * sizeof(uint64_t);
    return result;
}

uint64_t GpuProducer::host_generation_peak_bytes() const
{
    return host_borrowed_generation_peak_bytes() + impl_->capacity * sizeof(uint64_t);
}
uint64_t GpuProducer::host_borrowed_generation_peak_bytes() const
{
    auto const retained = impl_->retained() -
        (impl_->transfer ? impl_->transfer_capacity() * sizeof(uint64_t) : 0);
    auto const planned = retained + (impl_->output_bytes() ? 0 : impl_->capacity * sizeof(uint64_t));
    if (impl_->pool)
    {
        // run_pipeline borrows only pinned slot zero here. The production
        // three-slot writer queue and its index live outside this adapter;
        // the grow engine accounts for its own buffers separately.
        return planned;
    }
    // Production host estimates include rotating drains and compression/index
    // scratch. Keep that conservative bound for borrowed generation as well.
    return std::max(host_peak(impl_->k, impl_->selected), planned);
}
uint64_t GpuProducer::host_tiled_generation_peak_bytes() const
{
    // generate_into releases the full-size output before allocating its window.
    // Keep every other retained allocation, and count the window exactly once.
    auto const window = impl_->transfer_capacity() * sizeof(uint64_t);
    auto const retained = impl_->retained() - impl_->output_bytes() - (impl_->transfer ? window : 0);
    if (impl_->pool) return retained + window;
    // The production bound also includes host-sort buffers and cached streaming
    // tables. Do not subtract them or claim the device-sort output saving for
    // the tiny/pinned host-sort paths.
    return std::max(host_peak(impl_->k, impl_->selected), retained) + window;
}
uint64_t GpuProducer::retained_host_bytes() const { return impl_->retained(); }
uint64_t GpuProducer::runtime_host_allowance_bytes() noexcept { return runtime_host_bytes; }
uint64_t GpuProducer::device_resident_bytes() const
{
    auto const held = pos2gpu::sycl_backend::twophase_bytes_held(*impl_->queue);
    auto const* p = impl_->pool.get();
    if (!p) return held;
    return held + (impl_->device_output ? impl_->capacity * sizeof(uint64_t) : 0) +
        (impl_->device_packed ? impl_->packed_scratch_words * sizeof(uint64_t) : 0) +
        (p->d_storage ? p->storage_bytes : 0) + (p->d_pair_a ? p->pair_a_bytes : 0) +
        (p->d_pair_b ? p->pair_b_bytes : 0) + (p->d_sort_scratch ? p->sort_scratch_bytes : 0) +
        (p->d_counter ? sizeof(uint64_t) : 0);
}
std::string const& GpuProducer::tier() const { return impl_->selected; }
bool GpuProducer::profile_enabled() const { return impl_->profile; }

GpuProducer::BorrowedFragments::BorrowedFragments(
    std::span<uint64_t const> fragments, std::atomic<bool>& borrowed)
    : fragments_(fragments), borrowed_(&borrowed)
{
    borrowed_->store(true, std::memory_order_release);
}
GpuProducer::BorrowedFragments::~BorrowedFragments()
{
    if (borrowed_) borrowed_->store(false, std::memory_order_release);
}
GpuProducer::BorrowedFragments::BorrowedFragments(BorrowedFragments&& other) noexcept
    : fragments_(std::exchange(other.fragments_, {})),
      borrowed_(std::exchange(other.borrowed_, nullptr)) {}
GpuProducer::BorrowedFragments& GpuProducer::BorrowedFragments::operator=(BorrowedFragments&& other) noexcept
{
    if (this != &other)
    {
        if (borrowed_) borrowed_->store(false, std::memory_order_release);
        fragments_ = std::exchange(other.fragments_, {});
        borrowed_ = std::exchange(other.borrowed_, nullptr);
    }
    return *this;
}

GpuProducer::BorrowedFragments GpuProducer::generate_borrowed(std::array<uint8_t, 32> const& plot_id)
{
    impl_->check_generation();
    OutputWallTimings wall(impl_->profile, "generation-borrowed", impl_->next_profile_member());
    wall.slot_wait_ns += impl_->finish_output(wall.member);
    BorrowedFragments borrowed({}, impl_->handoff.borrowed());
    impl_->release(impl_->transfer);
    auto const config = impl_->config(plot_id);
    pos2gpu::GpuPipelineResult result;
    if (impl_->pool)
        result = pos2gpu::run_gpu_pipeline(config, *impl_->pool, 0);
    else
    {
        auto const scratch = impl_->scratch();
        result = pos2gpu::run_gpu_pipeline_streaming(config, impl_->output.get(), impl_->capacity, scratch);
    }
    auto const fragments = result.fragments();
    if (fragments.size() > impl_->capacity)
        throw std::runtime_error("GPU producer fragment count exceeds the admitted bound");
    if (!result.t3_fragments_storage.empty())
        throw std::logic_error("GPU producer borrowed output unexpectedly owns its fragments");
    borrowed.fragments_ = fragments;
    wall.complete = true;
    return borrowed;
}
uint64_t GpuProducer::generate_into(std::array<uint8_t, 32> const& plot_id,
    std::function<void(std::span<uint64_t const>, uint64_t)> const& consume)
{
    impl_->check_generation();
    if (!consume) throw std::invalid_argument("GPU producer requires an output consumer");
    OutputWallTimings wall(impl_->profile, "generation-tiled", impl_->next_profile_member());
    wall.slot_wait_ns += impl_->finish_output(wall.member);
    BorrowedFragments const lease({}, impl_->handoff.borrowed());
    impl_->prepare_tiled_output();
    using Consumer = std::function<void(std::span<uint64_t const>, uint64_t)>;
    struct Context { Consumer const& consume; uint64_t capacity; OutputWallTimings& wall; }
        context{consume, impl_->capacity, wall};
    pos2gpu::GpuFragmentSink sink;
    sink.pinned_window = {impl_->transfer.get(), impl_->transfer_capacity()};
    sink.context = &context;
    sink.consume = [](void* opaque, std::span<uint64_t const> tile, uint64_t total)
    {
        auto const& context = *static_cast<Context const*>(opaque);
        if (total > context.capacity)
            throw std::runtime_error("GPU producer fragment count exceeds the admitted bound");
        AddWallTime const timing(context.wall.consume_ns, context.wall.enabled);
        context.wall.bytes += tile.size_bytes();
        context.consume(tile, total);
    };
    auto const config = impl_->config(plot_id);
    auto const result = impl_->pool ? pos2gpu::run_gpu_pipeline(config, *impl_->pool, sink) :
        pos2gpu::run_gpu_pipeline_streaming(config, sink, impl_->scratch(false));
    if (!result.fragments().empty() || result.t3_count > impl_->capacity)
        throw std::logic_error("GPU producer tiled output returned invalid fragment storage or count");
    wall.complete = true;
    return result.t3_count;
}
bool GpuProducer::member_overlap_available() const { return impl_->device_output != nullptr; }
void GpuProducer::wait_for_output() { (void)impl_->finish_output(); }
std::shared_future<uint64_t> GpuProducer::generate_into_async(
    std::array<uint8_t, 32> const& plot_id,
    std::function<void(std::span<uint64_t const>, uint64_t)> consume)
{
    impl_->check_generation();
    if (!consume) throw std::invalid_argument("GPU producer requires an output consumer");
    return generate_output_async(plot_id, std::move(consume), {});
}
bool GpuProducer::compact_transfer_available() const { return impl_->device_packed != nullptr; }
std::shared_future<uint64_t> GpuProducer::generate_packed_into_async(
    std::array<uint8_t, 32> const& plot_id,
    std::function<void(PackedFragmentTile const&, uint64_t)> consume)
{
    impl_->check_generation();
    if (!consume) throw std::invalid_argument("GPU producer requires a packed output consumer");
    if (!compact_transfer_available()) throw std::logic_error("GPU producer compact transfer is unavailable");
    return generate_output_async(plot_id, {}, std::move(consume));
}
std::shared_future<uint64_t> GpuProducer::generate_output_async(
    std::array<uint8_t, 32> const& plot_id,
    std::function<void(std::span<uint64_t const>, uint64_t)> consume,
    std::function<void(PackedFragmentTile const&, uint64_t)> consume_packed)
{
    impl_->check_generation();
    if (!member_overlap_available())
        throw std::logic_error("GPU producer member overlap is unavailable");
    OutputWallTimings wall(impl_->profile, "generation-device-output", impl_->next_profile_member());
    BorrowedFragments const lease({}, impl_->handoff.borrowed());
    if (!impl_->transfer)
    {
        wall.slot_wait_ns += impl_->finish_output(wall.member);
        impl_->prepare_tiled_output();
    }
    pos2gpu::GpuDeviceFragmentOutput output;
    struct Context { Impl& impl; OutputWallTimings& wall; } context{*impl_, wall};
    output.context = &context;
    output.acquire = [](void* opaque, uint64_t count)
    {
        auto& context = *static_cast<Context*>(opaque);
        auto& impl = context.impl;
        if (count > impl.capacity)
            throw std::runtime_error("GPU producer fragment count exceeds the admitted bound");
        context.wall.slot_wait_ns += impl.finish_output(context.wall.member);
        return std::span<uint64_t>{impl.device_output, impl.capacity};
    };
    try
    {
        auto const result = pos2gpu::run_gpu_pipeline(impl_->config(plot_id), *impl_->pool, output);
        if (pos2gpu::async_error_count())
            throw std::runtime_error("GPU member generation reported an asynchronous backend error: " +
                                     pos2gpu::first_async_error());
        if (!result.fragments().empty() || result.t3_count > impl_->capacity)
            throw std::logic_error("GPU producer device output returned invalid fragment storage or count");
        auto pending = impl_->handoff.launch([impl = impl_.get(), count = result.t3_count, member = wall.member,
                                      consume = std::move(consume), consume_packed = std::move(consume_packed)]() mutable
        {
            OutputWallTimings wall(impl->profile, consume_packed ? "output-packed" : "output-raw", member);
            wall.observes_copy_wait = true;
            auto& queue = *impl->copy_queue;
            if (consume_packed)
            {
                try
                {
                    detail::PackedTransferLayout const layout(impl->k, count);
                    if (layout.metadata_words > impl->packed_scratch_words ||
                        layout.metadata_words > impl->transfer_capacity() / 2)
                        throw std::logic_error("GPU compact metadata exceeds the admitted transfer window");
                    (void)detail::launch_validate_packed_fragments(impl->k, impl->device_output, count,
                                                                  impl->device_packed, queue);
                    TimedCopyEvent metadata{queue.memcpy(impl->transfer.get(), impl->device_packed,
                        layout.metadata_words * sizeof(uint64_t)), wall, impl->copy_event_profiling};
                    wall.bytes += layout.metadata_words * sizeof(uint64_t);
                    metadata.wait_and_throw();
                    PackedMemberValidation validation;
                    auto const* summaries = reinterpret_cast<PackedMemberValidation const*>(
                        impl->transfer.get() + layout.boundary_words);
                    for (uint64_t group = 0; group < layout.validation_groups; ++group)
                    {
                        uint64_t const expected = std::min(detail::packed_values_per_group,
                            count - group * detail::packed_values_per_group);
                        auto const summary = summaries[group];
                        if (summary.checked_count != expected || (summary.flags & ~PackedMemberValidation::known_flags))
                            throw std::runtime_error("GPU compact validation coverage or flags differ");
                        validation.checked_count += summary.checked_count;
                        validation.flags |= summary.flags;
                    }
                    if (validation.checked_count != count ||
                        (validation.flags & (PackedMemberValidation::out_of_range | PackedMemberValidation::unsorted)))
                        throw std::runtime_error("GPU compact fragments are unsorted or out of range");
                    if (pos2gpu::async_error_count())
                        throw std::runtime_error("GPU compact validation reported an asynchronous backend error: " +
                                                 pos2gpu::first_async_error());
                    {
                        AddWallTime const timing(wall.consume_ns, wall.enabled);
                        consume_packed({{reinterpret_cast<uint32_t const*>(impl->transfer.get()),
                            size_t(layout.boundary_count)}, {}, validation}, count);
                    }
                    struct Context
                    {
                        std::function<void(PackedFragmentTile const&, uint64_t)> const& consume;
                        uint64_t count;
                        OutputWallTimings& wall;
                    } context{consume_packed, count, wall};
                    pos2gpu::GpuFragmentSink sink;
                    sink.pinned_window = {impl->transfer.get(), impl->transfer_capacity()};
                    sink.context = &context;
                    sink.consume = [](void* opaque, std::span<uint64_t const> words, uint64_t)
                    {
                        auto const& context = *static_cast<Context const*>(opaque);
                        if (words.empty()) return; // Empty members completed in the prelude.
                        AddWallTime const timing(context.wall.consume_ns, context.wall.enabled);
                        context.consume({{}, words, {}}, context.count);
                    };
                    sink.drain_overlapped(layout.payload_words, [&](std::span<uint64_t> tile, uint64_t offset)
                    {
                        if (tile.size() > impl->packed_scratch_words)
                            throw std::logic_error("GPU compact payload exceeds the admitted device tile");
                        (void)detail::launch_pack_fragment_words(impl->k, impl->device_output, count, offset,
                                                               tile.size(), impl->device_packed, queue);
                        AddWallTime const timing(wall.submit_ns, wall.enabled);
                        auto event = queue.memcpy(tile.data(), impl->device_packed, tile.size_bytes());
                        wall.bytes += tile.size_bytes();
                        return TimedCopyEvent{std::move(event), wall, impl->copy_event_profiling};
                    }, [&] { queue.wait_and_throw(); });
                    if (pos2gpu::async_error_count())
                        throw std::runtime_error("GPU compact output reported an asynchronous backend error: " +
                                                 pos2gpu::first_async_error());
                    wall.complete = true;
                    return count;
                }
                catch (...)
                {
                    try { queue.wait_and_throw(); } catch (...) {}
                    throw;
                }
            }
            using Consumer = std::function<void(std::span<uint64_t const>, uint64_t)>;
            struct Context { Consumer const& consume; OutputWallTimings& wall; } context{consume, wall};
            pos2gpu::GpuFragmentSink sink;
            sink.pinned_window = {impl->transfer.get(), impl->transfer_capacity()};
            sink.context = &context;
            sink.consume = [](void* opaque, std::span<uint64_t const> tile, uint64_t total)
            {
                auto& context = *static_cast<Context*>(opaque);
                AddWallTime const timing(context.wall.consume_ns, context.wall.enabled);
                context.consume(tile, total);
            };
            // The final sort is complete. This distinct queue shares its
            // context/device with generation but cannot order next Xs behind
            // the copies. Sink cleanup drains every copy before task completion
            // releases either the device slot or the pinned windows.
            sink.drain_overlapped(count, [&](std::span<uint64_t> tile, uint64_t offset)
            {
                AddWallTime const timing(wall.submit_ns, wall.enabled);
                auto event = queue.memcpy(tile.data(), impl->device_output + offset, tile.size_bytes());
                wall.bytes += tile.size_bytes();
                return TimedCopyEvent{std::move(event), wall, impl->copy_event_profiling};
            }, [&] { queue.wait_and_throw(); });
            // The production handler records asynchronous errors rather than
            // throwing from the callback. Never publish completion after one.
            if (pos2gpu::async_error_count())
                throw std::runtime_error("GPU member output reported an asynchronous backend error: " +
                                         pos2gpu::first_async_error());
            wall.complete = true;
            return count;
        });
        wall.complete = true;
        return pending;
    }
    catch (...)
    {
        // A failed next generation may coexist with the prior consumer. Wait
        // both queues and that consumer before exposing the failure or reuse.
        // Preserve the original generation/callback error if cleanup also fails.
        try { impl_->queue->wait_and_throw(); } catch (...) {}
        try { impl_->handoff.finish(); } catch (...) {}
        try { impl_->copy_queue->wait_and_throw(); } catch (...) {}
        throw;
    }
}
std::vector<uint64_t> GpuProducer::generate(std::array<uint8_t, 32> const& plot_id)
{
    auto const borrowed = generate_borrowed(plot_id);
    auto const fragments = borrowed.fragments();
    std::vector<uint64_t> owned(fragments.begin(), fragments.end());
    if (owned.capacity() > impl_->capacity)
        throw std::runtime_error("GPU producer output capacity exceeds the admitted fragment bound");
    return owned;
}
} // namespace group_grow
