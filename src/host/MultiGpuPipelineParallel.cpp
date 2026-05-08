// MultiGpuPipelineParallel.cpp — Phase 2.1c orchestrator. See header.

#include "host/MultiGpuPipelineParallel.hpp"

#include "gpu/SyclBackend.hpp"
#include "host/PoolSizing.hpp"

#include <sycl/sycl.hpp>

#include <stdexcept>
#include <thread>

namespace pos2gpu {

namespace {

// Buffer set for the T2 boundary. Allocated on the orchestrator
// thread; both worker threads read/write through these pointers. The
// first device's queue is used for the malloc_host so the buffers
// land in a shared pinned-host pool — on CUDA that's portable across
// all devices in the same process by default.
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
    GpuPipelineConfig const& cfg,
    int                      device_first,
    int                      device_second)
{
    if (cfg.k < 18 || cfg.k > 32 || (cfg.k & 1) != 0) {
        throw std::runtime_error("k must be even in [18, 32]");
    }
    if (cfg.strength < 2) {
        throw std::runtime_error("strength must be >= 2");
    }
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
                scratch.tiny_mode          = true;
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
                scratch.tiny_mode         = true;
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

} // namespace pos2gpu
