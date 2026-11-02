// Host-only checks for the bounded GPU output handoff; no device is initialized.
#include "host/GroupGpu.hpp"
#include "host/GpuPipeline.hpp"
#include "host/PoolSizing.hpp"
#include "host/VramBudget.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{
void require(bool condition, char const* message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename Action>
void rejects(Action action)
{
    try { action(); }
    catch (std::runtime_error const&) { return; }
    catch (std::invalid_argument const&) { return; }
    catch (std::logic_error const&) { return; }
    throw std::runtime_error("expected output handoff failure");
}

struct Consumer
{
    std::vector<uint64_t> received;
    std::vector<size_t> sizes;
    uint64_t const* window = nullptr;
    uint64_t total = 0;
    bool fail = false;

    static void append(void* opaque, std::span<uint64_t const> tile, uint64_t total)
    {
        auto& self = *static_cast<Consumer*>(opaque);
        require(total == self.total, "output count changed between tiles");
        require(tile.empty() || tile.data() == self.window, "output did not reuse the pinned window");
        self.sizes.push_back(tile.size());
        if (self.fail) throw std::runtime_error("consumer failed");
        self.received.insert(self.received.end(), tile.begin(), tile.end());
    }
};

// A transfer writes only when its event is waited or the queue is drained.
// This makes premature consumption, concurrent copies and early window reuse
// visible without creating a GPU context or relying on thread timing.
struct DeferredTransfers
{
    struct Copy
    {
        std::span<uint64_t> tile;
        uint64_t offset;
        bool ready = false, released = false;
    };
    struct Event
    {
        DeferredTransfers& owner;
        size_t index;
        void wait_and_throw() const
        {
            owner.complete(index);
            if (index == owner.fail_wait) throw std::runtime_error("event failed");
        }
    };
    std::span<uint64_t const> source;
    std::vector<Copy> copies;
    std::vector<uint64_t> received;
    size_t callbacks = 0, drains = 0;
    size_t fail_submit = SIZE_MAX, fail_wait = SIZE_MAX, fail_consume = SIZE_MAX;
    bool overlap_observed = false, fail_drain = false;

    explicit DeferredTransfers(std::span<uint64_t const> values) : source(values) {}

    Event submit(std::span<uint64_t> tile, uint64_t offset)
    {
        require(offset <= source.size() && tile.size() <= source.size() - offset,
                "deferred transfer exceeds source bounds");
        for (auto const& copy : copies)
        {
            require(copy.ready, "more than one D2H copy is in flight");
            require(copy.released || tile.data() + tile.size() <= copy.tile.data() ||
                        copy.tile.data() + copy.tile.size() <= tile.data(),
                    "window reused before its consumer finished");
        }
        size_t const index = copies.size();
        copies.push_back({tile, offset});
        // Model a submission that queued work but failed to return its event.
        if (index == fail_submit) throw std::runtime_error("submission failed");
        return {*this, index};
    }
    void complete(size_t index)
    {
        auto& copy = copies[index];
        if (!copy.ready)
            std::copy_n(source.begin() + copy.offset, copy.tile.size(), copy.tile.begin());
        copy.ready = true;
    }
    void drain()
    {
        ++drains;
        for (size_t i = 0; i < copies.size(); ++i)
        {
            complete(i);
            copies[i].released = true;
        }
        if (fail_drain) throw std::runtime_error("queue failed");
    }
    static void consume(void* opaque, std::span<uint64_t const> tile, uint64_t total)
    {
        auto& self = *static_cast<DeferredTransfers*>(opaque);
        require(total == self.source.size(), "deferred output count changed");
        if (tile.empty())
        {
            require(!total && self.copies.empty(), "invalid empty deferred output");
            ++self.callbacks;
            return;
        }
        auto& copy = self.copies[self.callbacks];
        require(copy.ready && tile.data() == copy.tile.data() && tile.size() == copy.tile.size(),
                "consumer read an unfinished or out-of-order transfer");
        if (self.callbacks + 1 < self.copies.size())
        {
            require(!self.copies[self.callbacks + 1].ready, "next transfer did not overlap consumption");
            self.overlap_observed = true;
        }
        size_t const index = self.callbacks++;
        if (index == self.fail_consume) throw std::runtime_error("consumer failed");
        self.received.insert(self.received.end(), tile.begin(), tile.end());
        copy.released = true;
    }
};

void overlapped_output(std::span<uint64_t const> source)
{
    std::array<uint64_t, 6> window{};
    auto run = [](std::span<uint64_t> allocation, DeferredTransfers& transfers)
    {
        pos2gpu::GpuFragmentSink sink{allocation, &transfers, DeferredTransfers::consume};
        sink.drain_overlapped(transfers.source.size(),
            [&](std::span<uint64_t> tile, uint64_t offset) { return transfers.submit(tile, offset); },
            [&] { transfers.drain(); });
    };
    DeferredTransfers ordered{source};
    run(window, ordered);
    require(std::equal(source.begin(), source.end(), ordered.received.begin(), ordered.received.end()) &&
                ordered.overlap_observed && !ordered.drains, "overlapped output changed data or failed to overlap");
    require(ordered.copies.back().tile.size() < ordered.copies.front().tile.size(),
            "overlapped output lost the short final tile");

    DeferredTransfers odd{source};
    run(std::span(window).first(window.size() - 1), odd);
    require(std::equal(source.begin(), source.end(), odd.received.begin(), odd.received.end()) && odd.overlap_observed,
            "odd transfer allocation changed output");
    DeferredTransfers single{source};
    run(std::span(window).first(1), single);
    require(std::equal(source.begin(), source.end(), single.received.begin(), single.received.end()) &&
                !single.overlap_observed, "single-element window fallback changed output");
    DeferredTransfers empty{source.first(0)};
    run(window, empty);
    require(empty.callbacks == 1 && empty.copies.empty() && !empty.drains,
            "empty overlapped output submitted a transfer");

    for (auto failure : {"consumer", "submission", "event"})
    {
        DeferredTransfers failed{source};
        if (std::string_view(failure) == "consumer") failed.fail_consume = 0;
        else if (std::string_view(failure) == "submission") failed.fail_submit = 1;
        else failed.fail_wait = 0;
        rejects([&] { run(window, failed); });
        require(failed.drains == 1 && !failed.copies.empty() &&
                    std::all_of(failed.copies.begin(), failed.copies.end(),
                                [](auto const& copy) { return copy.ready && copy.released; }),
                "failed overlapped output left an unfinished transfer");
        if (std::string_view(failure) == "consumer")
            require(failed.overlap_observed, "consumer failure did not exercise a pending transfer");
        DeferredTransfers reused{source};
        run(window, reused);
        require(std::equal(source.begin(), source.end(), reused.received.begin(), reused.received.end()),
                "transfer windows could not be reused after failure");
    }
    DeferredTransfers primary{source};
    primary.fail_consume = 0; primary.fail_drain = true;
    try { run(window, primary); throw std::logic_error("expected consumer failure"); }
    catch (std::runtime_error const& error)
    { require(std::string_view(error.what()) == "consumer failed", "cleanup replaced the primary output error"); }
    require(std::all_of(primary.copies.begin(), primary.copies.end(),
                        [](auto const& copy) { return copy.ready; }), "cleanup error left a pending transfer");
}

void member_output(std::span<uint64_t const> source)
{
    group_grow::detail::OutputHandoff handoff;
    std::vector<uint64_t> device_slot(source.begin(), source.end());
    std::array<uint64_t, 6> window{};
    DeferredTransfers transfers{device_slot};
    std::promise<void> entered, release, next_generated;
    auto entered_future = entered.get_future();
    auto release_future = release.get_future();
    auto next_generated_future = next_generated.get_future();
    bool reentry_rejected = false;
    struct Context
    {
        DeferredTransfers& transfers;
        std::promise<void>& entered;
        std::future<void>& release;
    } context{transfers, entered, release_future};
    auto pending = handoff.launch([&]
    {
        try { handoff.check_generation(); }
        catch (std::logic_error const&) { reentry_rejected = true; }
        pos2gpu::GpuFragmentSink sink{window, &context,
            [](void* opaque, std::span<uint64_t const> tile, uint64_t total)
            {
                auto& context = *static_cast<Context*>(opaque);
                if (!context.transfers.callbacks)
                {
                    context.entered.set_value();
                    context.release.get();
                }
                DeferredTransfers::consume(&context.transfers, tile, total);
            }};
        sink.drain_overlapped(source.size(),
            [&](std::span<uint64_t> tile, uint64_t offset) { return transfers.submit(tile, offset); },
            [&] { transfers.drain(); });
        return uint64_t(source.size());
    });
    std::future<void> controller;
    struct DrainOnExit
    {
        std::promise<void>& release;
        std::promise<void>& next_generated;
        std::shared_future<uint64_t>& pending;
        std::future<void>& controller;
        ~DrainOnExit()
        {
            // Assertions before the controller starts must release the blocked
            // consumer; assertions after it starts must also wake its gate.
            try { release.set_value(); } catch (...) {}
            try { next_generated.set_value(); } catch (...) {}
            if (pending.valid()) pending.wait();
            if (controller.valid()) controller.wait();
        }
    } drain_on_exit{release, next_generated, pending, controller};
    require(entered_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
            "member output did not enter its blocked consumer");
    entered_future.get();
    // Generation is allowed while consumption is blocked; borrowing and a
    // second drain task are not. This tests the actual producer's slot gate.
    handoff.check_generation();
    handoff.borrowed().store(true, std::memory_order_release);
    rejects([&] { handoff.check_generation(); });
    handoff.borrowed().store(false, std::memory_order_release);
    handoff.check_generation();
    rejects([&] { handoff.launch([] { return uint64_t{}; }); });
    require(pending.wait_for(std::chrono::seconds(0)) == std::future_status::timeout,
            "blocked output completed before the next member generated");

    struct Destination
    {
        group_grow::detail::OutputHandoff& handoff;
        std::span<uint64_t> slot;
        std::promise<void>& next_generated;
    } destination{handoff, device_slot, next_generated};
    pos2gpu::GpuDeviceFragmentOutput output{&destination,
        [](void* opaque, uint64_t)
        {
            auto& destination = *static_cast<Destination*>(opaque);
            destination.next_generated.set_value();
            destination.handoff.finish();
            return destination.slot;
        }};
    controller = std::async(std::launch::async, [&]
    {
        next_generated_future.get();
        bool const blocked = pending.wait_for(std::chrono::seconds(0)) == std::future_status::timeout;
        release.set_value();
        require(blocked, "slot acquisition did not preserve the pending consumer");
    });
    auto const reused = output.destination(source.size());
    controller.get();
    require(pending.get() == source.size() && reentry_rejected && reused.data() == device_slot.data(),
            "member output count, thread ownership or slot reuse changed");
    require(std::equal(source.begin(), source.end(), transfers.received.begin(), transfers.received.end()),
            "member output was overwritten before its consumer finished");
    std::fill(reused.begin(), reused.end(), uint64_t{});
    require(std::equal(source.begin(), source.end(), transfers.received.begin(), transfers.received.end()),
            "completed member still borrowed the device destination");

    DeferredTransfers failed{source};
    failed.fail_consume = 0;
    auto failure = handoff.launch([&]
    {
        pos2gpu::GpuFragmentSink sink{window, &failed, DeferredTransfers::consume};
        sink.drain_overlapped(source.size(),
            [&](std::span<uint64_t> tile, uint64_t offset) { return failed.submit(tile, offset); },
            [&] { failed.drain(); });
        return uint64_t(source.size());
    });
    rejects([&] { handoff.finish(); });
    rejects([&] { (void)failure.get(); });
    require(failed.drains == 1 && std::all_of(failed.copies.begin(), failed.copies.end(),
                [](auto const& copy) { return copy.ready && copy.released; }),
            "member consumer error escaped before its transfers drained");
    auto recovered = handoff.launch([&] { return uint64_t(source.size()); });
    handoff.finish();
    require(recovered.get() == source.size(), "failed member prevented drained slot reuse");

    pos2gpu::GpuDeviceFragmentOutput no_acquire{};
    rejects([&] { no_acquire.destination(source.size()); });
    pos2gpu::GpuDeviceFragmentOutput short_output{&device_slot,
        [](void* opaque, uint64_t)
        {
            auto& slot = *static_cast<std::vector<uint64_t>*>(opaque);
            return std::span<uint64_t>(slot).first(1);
        }};
    rejects([&] { short_output.destination(source.size()); });
    pos2gpu::GpuDeviceFragmentOutput empty_output{nullptr,
        [](void*, uint64_t) { return std::span<uint64_t>{}; }};
    rejects([&] { empty_output.destination(0); });
}

void automatic_overlap()
{
    // Optional output must not consume the scratch that made the serial
    // two-phase match admissible. Exercise both exact boundaries and a serial
    // path that was already using allocation-free matching.
    constexpr uint64_t margin = 128;
    constexpr uint64_t required = 1024;
    for (int k = 18; k <= 28; k += 2) for (int strength : {2, k - 3})
    {
        auto const section_bits = k < 28 ? 2 : k - 26;
        auto const capacity = pos2gpu::match_phase_capacity(k, section_bits);
        auto const buckets = uint64_t(1) << (section_bits + strength);
        auto const scratch = pos2gpu::twophase_scratch_bytes(capacity, buckets);
        auto const output = capacity * sizeof(uint64_t);
        auto const free = required + margin + output + scratch;
        auto const serial = pos2gpu::vram_scratch_budget(free, required, margin);
        auto const overlapped = pos2gpu::vram_scratch_budget(free, required + output, margin);
        require(group_grow::detail::preserves_match_scratch(serial, overlapped, scratch),
                "exact overlap admission lost its fast match allocation");
        require(!group_grow::detail::preserves_match_scratch(serial - 1, overlapped - 1, scratch),
                "automatic overlap evicted an admitted fast match allocation");
        require(group_grow::detail::preserves_match_scratch(scratch - 1, 0, scratch),
                "automatic overlap rejected an already allocation-free match path");
        require(pos2gpu::twophase_candidate_capacity(capacity, 0) ==
                    pos2gpu::twophase_candidate_capacity(capacity, 1),
                "empty kernel bucket range changed candidate sizing");
    }
}

void output_destruction()
{
    for (bool fail : {false, true})
    {
        std::promise<void> entered, release, destroying;
        auto entered_future = entered.get_future();
        auto release_future = release.get_future();
        auto destroying_future = destroying.get_future();
        std::shared_future<uint64_t> pending;
        bool completed = false;
        std::future<void> controller;
        {
            group_grow::detail::OutputHandoff handoff;
            struct UnblockOnFailure
            {
                std::promise<void>& destroying;
                std::promise<void>& release;
                bool active = true;
                ~UnblockOnFailure()
                {
                    if (!active) return;
                    try { destroying.set_value(); } catch (...) {}
                    try { release.set_value(); } catch (...) {}
                }
            } unblock{destroying, release};
            pending = handoff.launch([&]
            {
                entered.set_value();
                release_future.get();
                completed = true;
                if (fail) throw std::runtime_error("consumer failed on destruction");
                return uint64_t{};
            });
            entered_future.get();
            controller = std::async(std::launch::async, [&]
            {
                destroying_future.get();
                bool const blocked = pending.wait_for(std::chrono::seconds(0)) == std::future_status::timeout;
                release.set_value();
                require(blocked, "output consumer finished before owner destruction");
            });
            destroying.set_value();
            unblock.active = false;
        }
        controller.get();
        require(completed, "output owner released its state before consumer completion");
        if (fail) rejects([&] { (void)pending.get(); });
        else require(pending.get() == 0, "output owner changed the completed result");
    }
}
} // namespace

int main()
{
    static_assert(!std::is_copy_constructible_v<group_grow::GpuProducer::BorrowedFragments>);
    static_assert(std::is_nothrow_move_constructible_v<group_grow::GpuProducer::BorrowedFragments>);
    static_assert(std::is_nothrow_move_assignable_v<group_grow::GpuProducer::BorrowedFragments>);
    try
    {
        std::array<uint64_t, 3> window{};
        std::array<uint64_t, 8> const source{0, 0, 1, 3, 3, 5, 9, 12};
        Consumer consumer;
        consumer.window = window.data();
        consumer.total = source.size();
        pos2gpu::GpuFragmentSink sink{window, &consumer, Consumer::append};
        size_t transfers = 0;
        auto transfer = [&](std::span<uint64_t> tile, uint64_t offset)
        {
            require(offset <= source.size() && tile.size() <= source.size() - offset,
                    "transfer exceeds source bounds");
            ++transfers;
            std::copy_n(source.begin() + offset, tile.size(), tile.begin());
        };
        sink.drain(source.size(), transfer);
        require(std::equal(source.begin(), source.end(), consumer.received.begin(), consumer.received.end()),
                "tiled output changed fragment order or data");
        require(consumer.sizes == std::vector<size_t>{3, 3, 2}, "short final tile was not preserved");
        require(transfers == consumer.sizes.size(), "transfer/consume calls differ");

        consumer.received.clear();
        consumer.sizes.clear();
        consumer.total = 0;
        transfers = 0;
        sink.drain(0, transfer);
        require(transfers == 0 && consumer.sizes == std::vector<size_t>{0},
                "empty output must notify the consumer without a transfer");

        consumer.sizes.clear();
        consumer.total = source.size();
        consumer.fail = true;
        rejects([&] { sink.drain(source.size(), transfer); });
        require(transfers == 1 && consumer.sizes == std::vector<size_t>{3},
                "consumer exception did not stop output reuse");
        consumer.fail = false;
        consumer.sizes.clear();
        rejects([&]
        {
            sink.drain(source.size(), [](std::span<uint64_t>, uint64_t)
            { throw std::runtime_error("transfer failed"); });
        });
        require(consumer.sizes.empty(), "failed transfer reached the consumer");

        // A failed consumer or transfer leaves no delivery state behind.
        sink.drain(source.size(), transfer);
        require(std::equal(source.begin(), source.end(), consumer.received.begin(), consumer.received.end()),
                "output could not be reused after an exception");

        pos2gpu::GpuFragmentSink no_window{{}, &consumer, Consumer::append};
        rejects([&] { no_window.drain(source.size(), transfer); });
        pos2gpu::GpuFragmentSink no_consumer{window, &consumer, nullptr};
        rejects([&] { no_consumer.drain(source.size(), transfer); });
        overlapped_output(source);
        member_output(source);
        automatic_overlap();
        output_destruction();
        std::cout << "bounded GPU output handoff checks passed\n";
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
