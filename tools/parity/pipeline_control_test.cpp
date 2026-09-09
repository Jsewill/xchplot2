// Real coordinator, fake GPU execution. An alarm turns a deadlock into failure.
#undef NDEBUG
#include "host/MultiGpuPipelineParallel.hpp"
#include "host/GpuBufferPool.hpp"
#include "host/Cancel.hpp"
#include "host/VramBudget.hpp"
#include <atomic>
#include <cassert>
#include <iostream>
#include <unistd.h>

namespace {
thread_local int device = 0;
int failing_device = -1;
uint64_t free_vram = 8ULL << 30;
std::atomic<int> calls{0};
}
namespace pos2gpu {
void bind_current_device(int id) { device = id; }
void host_pinned_reserve_check(size_t, char const*) {}
DeviceMemInfo query_device_memory() { return {.free_bytes = free_vram, .total_bytes = 8ULL << 30}; }
size_t vram_safety_margin() { return 128ULL << 20; }
size_t streaming_peak_bytes(int k) { return streaming_base_peak_bytes(k, StreamingTier::Compact); }
size_t streaming_minimal_peak_bytes(int k) { return streaming_base_peak_bytes(k, StreamingTier::Minimal); }
size_t streaming_tiny_peak_bytes(int k) { return streaming_base_peak_bytes(k, StreamingTier::Tiny); }
GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const&, uint64_t* dst, size_t,
                                             StreamingPinnedScratch const&)
{
    ++calls;
    if (device == failing_device) throw std::runtime_error("injected stage failure");
    dst[0] = 42;
    GpuPipelineResult r;
    r.external_fragments_ptr = dst; r.external_fragments_count = 1;
    r.t1_count = r.t2_count = r.t3_count = 1;
    return r;
}
}
int main()
{
    using namespace pos2gpu;
    alarm(15);
    std::vector<GpuPipelineConfig> configs(5);
    for (auto& c : configs) c.k = 18;
    for (auto devices : {std::vector<int>{0, 1}, std::vector<int>{0, 1, 2}}) {
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
    alarm(0);
    std::cout << "Pipeline: stage/callback failures, cancellation, and VRAM rejection passed\n";
}
