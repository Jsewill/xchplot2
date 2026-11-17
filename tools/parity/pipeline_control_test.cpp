// Real coordinator, fake GPU execution. A timeout turns a deadlock into failure.
#undef NDEBUG
#include "host/MultiGpuPipelineParallel.hpp"
#include "host/MultiGpuShardBufferPool.hpp"
#include "host/GpuBufferPool.hpp"
#include "host/Cancel.hpp"
#include "host/VramBudget.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>

namespace {
thread_local int device = 0;
int failing_device = -1;
uint64_t free_vram = 8ULL << 30;
std::atomic<int> calls{0};
bool deny_host_reserve = false;
bool device_boundary = false;

void shard_pool_failures()
{
    sycl::queue q;
    for (int kind : {0, 1, 2}) {
        for (int failure : {0, 1, 2}) {
            if (failure == 0 && kind != 2) continue; // reserve check is host-only
            for (bool retry : {false, true}) {
                pos2gpu::ShardBufferPool pool(&q);
                auto acquire = [&](size_t count) -> uint64_t* {
                    if (kind == 0) return pool.ensure<uint64_t>("scratch", count);
                    if (kind == 1) return static_cast<uint64_t*>(pool.ensure_bytes("scratch", count * sizeof(uint64_t)));
                    return pool.ensure_host<uint64_t>("scratch", count);
                };
                acquire(8)[0] = 42;
                assert(pool.total_bytes() == 8 * sizeof(uint64_t));
                deny_host_reserve = failure == 0;
                sycl::fail_allocation = failure == 1 ? sycl::allocation_failure::null :
                    failure == 2 ? sycl::allocation_failure::exception : sycl::allocation_failure::none;
                bool threw = false;
                try { acquire(16); }
                catch (std::runtime_error const&) { threw = true; }
                deny_host_reserve = false;
                sycl::fail_allocation = sycl::allocation_failure::none;
                assert(threw && pool.total_bytes() == 0);
                if (retry) {
                    auto* next = acquire(8);
                    next[0] = 43;
                    assert(acquire(4) == next && next[0] == 43);
                    assert(pool.total_bytes() == 8 * sizeof(uint64_t));
                }
            }
        }
    }
    assert(sycl::allocations.empty());
}
}
namespace pos2gpu {
void bind_current_device(int id) { device = id; }
void host_pinned_reserve_check(size_t, char const*)
{
    if (deny_host_reserve) throw std::runtime_error("injected host reserve denial");
}
DeviceMemInfo query_device_memory() { return {.free_bytes = free_vram, .total_bytes = 8ULL << 30}; }
size_t vram_safety_margin() { return 128ULL << 20; }
size_t streaming_peak_bytes(int k) { return streaming_base_peak_bytes(k, StreamingTier::Compact); }
size_t streaming_minimal_peak_bytes(int k) { return streaming_base_peak_bytes(k, StreamingTier::Minimal); }
size_t streaming_tiny_peak_bytes(int k) { return streaming_base_peak_bytes(k, StreamingTier::Tiny); }
GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const&, uint64_t* dst, size_t,
                                             StreamingPinnedScratch const& scratch)
{
    ++calls;
    if (device == failing_device) throw std::runtime_error("injected stage failure");
    assert(!sycl::is_device_allocation(scratch.h_keys_merged));
    assert(sycl::is_device_allocation(scratch.h_t2_meta) == device_boundary);
    assert(sycl::is_device_allocation(scratch.h_t2_xbits) == device_boundary);
    dst[0] = 42;
    GpuPipelineResult r;
    r.external_fragments_ptr = dst; r.external_fragments_count = 1;
    r.t1_count = r.t2_count = r.t3_count = 1;
    return r;
}
}
int main(int argc, char** argv)
{
    using namespace pos2gpu;
    bool const peer_copy = argc == 2 && std::string_view(argv[1]) == "--peer-copy";
#ifdef _WIN32
    _putenv_s("POS2GPU_PIPELINE_PEER_COPY", peer_copy ? "1" : "0");
#else
    setenv("POS2GPU_PIPELINE_PEER_COPY", peer_copy ? "1" : "0", 1);
#endif
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::seconds{15});
        std::_Exit(1);
    }).detach();
    shard_pool_failures();
    std::vector<GpuPipelineConfig> configs(5);
    for (auto& c : configs) c.k = 18;
    for (auto devices : {std::vector<int>{0, 1}, std::vector<int>{0, 1, 2}}) {
        device_boundary = peer_copy && devices.size() == 2;
        for (int fail = -1; fail < static_cast<int>(devices.size()); ++fail) {
            reset_cancel_for_tests(); failing_device = fail; calls = 0;
            bool threw = false;
            try {
                auto r = run_pipeline_parallel_batch(configs, devices, 1);
                assert(r.size() == configs.size());
                for (auto const& result : r) assert(result.fragments_storage == std::vector<uint64_t>{42});
            } catch (std::runtime_error const&) { threw = true; }
            assert(threw == (fail >= 0));
        }
        failing_device = -1;
        bool threw = false;
        try {
            run_pipeline_parallel_batch(configs, devices, 1, {},
                [](int, PipelineParallelSplitResult) { throw std::runtime_error("writer callback failure"); });
        } catch (std::runtime_error const&) { threw = true; }
        assert(threw);
        request_cancel(); calls = 0;
        run_pipeline_parallel_batch(configs, devices, 1);
        assert(calls == 0);
        reset_cancel_for_tests();
        free_vram = 128ULL << 20; calls = 0;
        threw = false;
        try { run_pipeline_parallel_batch(configs, devices, 1); }
        catch (std::runtime_error const&) { threw = true; }
        assert(threw && calls == 0);
        free_vram = 8ULL << 30;
    }
    device_boundary = peer_copy;
    for (auto first : {PipelineStageTier::Tiny, PipelineStageTier::Minimal}) {
        for (auto second : {PipelineStageTier::Tiny, PipelineStageTier::Minimal}) {
            auto split = run_pipeline_parallel_split(configs.front(), {0, 1}, {first, second});
            assert(split.fragments_storage == std::vector<uint64_t>{42});
            auto batch = run_pipeline_parallel_batch(configs, {0, 1}, 2, {first, second});
            for (auto const& result : batch) assert(result.fragments_storage == std::vector<uint64_t>{42});
        }
    }
    assert(sycl::allocations.empty());
    std::cout << "Pipeline: failures, cancellation, memory rejection, pool recovery and boundary storage passed\n";
}
