// Reusable GPU producer for ordinary group construction.
#pragma once
#include "host/GroupPacked.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace group_grow
{
namespace detail
{
// Automatic overlap must not evict the admitted fast match path merely to
// retain a prior output. If the serial grant already cannot admit that path,
// spending its unused headroom on output does not change match selection.
inline bool preserves_match_scratch(uint64_t serial_grant, uint64_t overlap_grant,
                                    uint64_t required) noexcept
{
    return serial_grant < required || overlap_grant >= required;
}

// One completed device output may be consumed while its producer computes the
// next member. Like BatchPlotter's SlotGate, completion (not dequeue) releases
// the slot. Kept host-only so thread, borrow and failure ownership are testable.
class OutputHandoff
{
public:
    ~OutputHandoff() { if (pending_.valid()) pending_.wait(); }

    void check_thread() const
    {
        if (std::this_thread::get_id() != thread_)
            throw std::logic_error("GPU producer must be used on its constructing thread");
    }
    void check_generation() const
    {
        check_thread();
        if (borrowed_.load(std::memory_order_acquire))
            throw std::logic_error("GPU producer output is still borrowed by a consumer");
    }
    std::atomic<bool>& borrowed() noexcept { return borrowed_; }
    bool pending() const { check_thread(); return pending_.valid(); }
    void finish()
    {
        check_thread();
        // The task has finished even when get throws. Release its state so a
        // drained callback failure does not permanently poison slot reuse.
        auto pending = std::exchange(pending_, {});
        if (pending.valid()) (void)pending.get();
    }
    template <typename Drain>
    std::shared_future<uint64_t> launch(Drain&& drain)
    {
        check_thread();
        if (pending_.valid())
            throw std::logic_error("GPU producer output slot is still being consumed");
        pending_ = std::async(std::launch::async, std::forward<Drain>(drain)).share();
        return pending_;
    }

private:
    std::thread::id thread_ = std::this_thread::get_id();
    std::atomic<bool> borrowed_{false};
    std::shared_future<uint64_t> pending_;
};
} // namespace detail

struct GpuProducerOptions
{
    int device_id = 0;
    // Explicit resource offer. Zero preserves ordinary free-VRAM admission.
    uint64_t max_vram_bytes = 0;
    // Application host compute allowance. Runtime/driver threads are outside
    // this cap; zero preserves the ordinary producer's host merge policy.
    unsigned host_threads = 0;
    std::string tier = "auto";
    bool profile = false;
    // Internal opt-in from the growth command's generation-overlap policy.
    // Unsupported tiers or insufficient extra VRAM keep synchronous output.
    bool member_overlap = false;
    // Automatic policy preserves an admitted two-phase match allocation.
    // Explicit overlap may spend this headroom on the independent output.
    bool preserve_match_scratch = true;
    // Packed transport for compact consumers without a precomputed recovery
    // digest. Raw output and unsupported tiers/headroom keep their handoff.
    bool compact_transfer = false;
};

struct GpuProducerPreflight
{
    std::string tier;
    uint64_t borrowed_host_bytes = 0, owning_host_bytes = 0;
    uint64_t device_bytes = 0;
    std::string physical_identity;
};

// Construct, generate and destroy on the same thread. Device generation is
// synchronous; the optional bounded output consumer may run on another thread.
class GpuProducer
{
public:
    // Borrows the producer's output slot. Keep this handle and the producer
    // alive until every reader (including engine futures) has finished. No
    // generation may start while a handle is alive; releasing a handle from
    // the consuming thread is allowed. A moved-from handle has an empty span.
    class BorrowedFragments
    {
    public:
        BorrowedFragments() = default;
        ~BorrowedFragments();
        BorrowedFragments(BorrowedFragments const&) = delete;
        BorrowedFragments& operator=(BorrowedFragments const&) = delete;
        BorrowedFragments(BorrowedFragments&& other) noexcept;
        BorrowedFragments& operator=(BorrowedFragments&& other) noexcept;
        std::span<uint64_t const> fragments() const noexcept { return fragments_; }

    private:
        friend class GpuProducer;
        BorrowedFragments(std::span<uint64_t const> fragments, std::atomic<bool>& borrowed);
        std::span<uint64_t const> fragments_;
        std::atomic<bool>* borrowed_ = nullptr;
    };

    GpuProducer(int k, int strength, GpuProducerOptions options = {});
    // Uses the production backend's sizing queries and current free memory.
    // Initializes the device runtime, but allocates no generation tables.
    static GpuProducerPreflight preflight(int k, int strength, GpuProducerOptions const& = {});
    ~GpuProducer();
    GpuProducer(GpuProducer const&) = delete;
    GpuProducer& operator=(GpuProducer const&) = delete;

    // Constructor selects the production pool/streaming path and initializes
    // device state, but defers every pinned allocation until admitted generate().
    // Generation peak includes retained host buffers, runtime allowance, and
    // the returned vector. Count other resident members separately. Outside
    // generation, count retained_host_bytes() alongside every resident member.
    uint64_t host_generation_peak_bytes() const;
    // Same generation peak without an owning output copy. The borrowed span
    // is already included in the producer's retained host buffers.
    uint64_t host_borrowed_generation_peak_bytes() const;
    // Bounded output transfer; count the consumer's compact destination
    // separately. Existing host-sort scratch remains in the tier estimate.
    uint64_t host_tiled_generation_peak_bytes() const;
    uint64_t retained_host_bytes() const;
    // Live persistent device tables and cached two-phase match scratch;
    // runtime/context and transient allocations use production VRAM admission.
    uint64_t device_resident_bytes() const;
    // Queue/runtime state can outlive a failed auto-backend attempt. CPU
    // fallback must continue to reserve this allowance after GPU initialization.
    static uint64_t runtime_host_allowance_bytes() noexcept;
    std::string const& tier() const;
    bool profile_enabled() const;
    BorrowedFragments generate_borrowed(std::array<uint8_t, 32> const& plot_id);
    // Tiles borrow alternating subwindows of the admitted pinned allocation
    // only until consume returns. The next device transfer can run concurrently.
    // The consumer must finish reading each tile synchronously; total_count is
    // constant and permits allocating a compact destination before appending.
    // A prior full-size output slot is released before allocating the window.
    uint64_t generate_into(std::array<uint8_t, 32> const& plot_id,
        std::function<void(std::span<uint64_t const>, uint64_t total_count)> const& consume);
    bool member_overlap_available() const;
    // Requires member_overlap_available(). Owns one independent device output
    // slot and the existing transfer window. The next call may generate through
    // T3 match while this consumer runs, then waits for it before final sort
    // reuses the slot. Callbacks remain ordered, with one consumer active.
    // Capture consumer state by value and read it only after future.get(). A
    // callback must finish all tile reads and member/hash finalization before
    // returning. It must not call producer methods. The producer and captured
    // state remain alive until every returned future has completed.
    std::shared_future<uint64_t> generate_into_async(std::array<uint8_t, 32> const& plot_id,
        std::function<void(std::span<uint64_t const>, uint64_t total_count)> consume);
    bool compact_transfer_available() const;
    // Same output ownership and ordering as generate_into_async. The prelude
    // follows completed device validation; payload tiles borrow packed words.
    // Consumers precomputing the ordinary recovery digest must use raw output.
    std::shared_future<uint64_t> generate_packed_into_async(std::array<uint8_t, 32> const& plot_id,
        std::function<void(PackedFragmentTile const&, uint64_t total_count)> consume);
    // Waits for the active output consumer and propagates its exception. Serial
    // generation calls also wait. Destruction waits without throwing.
    void wait_for_output();
    // Owns its fragments and survives reuse or destruction of the producer.
    std::vector<uint64_t> generate(std::array<uint8_t, 32> const& plot_id);

private:
    std::shared_future<uint64_t> generate_output_async(std::array<uint8_t, 32> const& plot_id,
        std::function<void(std::span<uint64_t const>, uint64_t)> consume,
        std::function<void(PackedFragmentTile const&, uint64_t)> consume_packed);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace group_grow
